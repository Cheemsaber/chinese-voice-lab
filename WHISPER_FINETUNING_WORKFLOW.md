# Whisper Fine-Tuning Workflow

This guide describes a practical workflow for adapting Whisper to recognize device identifiers embedded in realistic mixed Chinese-English operating commands, for example:

```text
置 018VB 手动，MANU 018VB
```

The verification application does not need to interpret the entire command. It transcribes the speech, extracts identifiers such as `018VB`, extracts the corresponding identifier from the screen through OCR, normalizes both values, and compares them.

## Current hardware constraints

The development computer currently has:

- NVIDIA T550 Laptop GPU with 4 GB VRAM
- Approximately 64 GB system memory
- 16 logical CPU processors
- Python 3.9

Four gigabytes of VRAM is insufficient for practical local fine-tuning of Whisper Large. The recommended arrangement is therefore:

1. Prepare and validate the dataset locally.
2. Run a small-model training pilot locally.
3. Fine-tune Whisper Large-v3 with LoRA on a temporary 24 GB cloud GPU.
4. Export and quantize the trained model for local inference.

## 1. Define the recognition and verification pipeline

Keep speech recognition and identifier comparison as separate stages:

```text
Full speech -> Whisper transcript -> extract and normalize device identifiers
Screen image -> OCR text          -> extract and normalize device identifiers
                                             |
                                      compare identifiers
```

Whisper should normally produce the complete spoken sentence. It should not be trained to return only the device identifier when its input contains a full sentence. Keeping the full transcription makes errors visible and prevents the ASR model from silently learning to discard surrounding speech.

Use a strict extractor after transcription. For identifiers consisting of three digits followed by two uppercase letters, an initial pattern is:

```regex
(?<![A-Z0-9])\d{3}[A-Z]{2}(?![A-Z0-9])
```

Normalize identifiers before comparison by removing permitted separators and spaces and converting letters to uppercase. If a command contains two different valid identifiers, return an ambiguous or review-required result instead of selecting one automatically. Repeated instances of the same identifier may be deduplicated.

Do not provide the OCR result to Whisper as a decoding hint. The speech and screen readings should remain independent; otherwise, the system could be biased toward declaring a match.

## 2. Collect realistic audio

The training distribution should resemble actual operation:

- 70-80% complete operating sentences
- 15-20% shortened commands or phrases
- 5-10% identifier-only speech

Record examples containing identifiers at the beginning, middle, and end of sentences. Include normal and difficult delivery conditions:

- Normal, fast, and deliberately vague articulation
- Natural pauses and self-corrections
- Transitions between Chinese words, digits, and English letters
- Different operators, accents, microphones, rooms, and background noise
- Repeated identifiers
- Similar or easily confused letters such as `B/D/P/T`, `M/N`, `F/S`, and `G/J`
- Repeated digits and letters, such as `000AA`, `777BB`, and `101MN`
- Commands with no identifier, malformed identifiers, and conflicting repetitions

Store uncompressed mono PCM WAV audio at 16 kHz where possible. Retain the original recording even when additional training clips are cropped from it.

## 3. Create labels and metadata

Use full verbatim transcripts with identifiers written in one consistent canonical form:

```text
置 018VB 手动，MANU 018VB
```

Suggested manifest entry:

```json
{
  "audio": "audio/operator03/session02/000184.wav",
  "text": "置 018VB 手动，MANU 018VB",
  "device_ids": ["018VB"],
  "speaker_id": "operator03",
  "session_id": "session02",
  "condition": "fast_noisy"
}
```

Useful metadata includes:

- Speaker and recording-session identifiers
- Microphone or workstation identifier
- Room or noise condition
- Speaking speed and articulation category
- Expected device identifiers
- Whether the utterance is valid, invalid, ambiguous, or contains no identifier

Balance every digit and letter across every valid identifier position. Complete coverage of all `000AA` through `999ZZ` combinations is not necessary, but the individual symbols, difficult pairs, and positional transitions should be well represented.

## 4. Split the dataset safely

Create fixed training, validation, and test partitions before fine-tuning. An initial split can be 80/10/10.

Split by speaker and recording session rather than by individual audio file. A speaker or session reserved for evaluation must not appear in training. Keep an original recording and every crop or augmentation derived from it in the same partition to prevent data leakage.

The locked test set should include:

- Speakers and sessions not used for training
- Fast and indistinct speech
- Real operating noise
- Near-confusable identifiers
- Matching and deliberately mismatching speech/OCR pairs
- Commands containing zero, one, repeated, and conflicting identifiers

Do not apply artificial augmentation to the validation or test recordings.

## 5. Establish the baseline

Run the current recognition and extraction pipeline over the locked test set before fine-tuning. Save the raw transcript, extracted identifiers, expected identifiers, and verification decision for every sample.

Measure at least:

- Exact identifier accuracy
- Per-character accuracy
- Identifier detection recall
- Invalid or ambiguous rejection rate
- False-reject rate for genuine matches
- False-accept rate when the spoken and displayed identifiers differ

Exact identifier accuracy and false-accept rate are the primary metrics. Ordinary word error rate is useful for diagnosing the surrounding sentence, but it should not select the production checkpoint.

## 6. Run a local small-model pilot

Use the local 4 GB GPU to verify the dataset loader, tokenizer, collator, metrics, checkpointing, and export path. Start with the multilingual model:

```text
openai/whisper-base
```

Suggested pilot configuration:

```yaml
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
gradient_checkpointing: true
fp16: true

lora_rank: 8
lora_alpha: 16
lora_dropout: 0.05
target_modules:
  - q_proj
  - v_proj

language: Chinese
task: transcribe
```

Prefer clips shorter than 10-15 seconds for this pilot. Whisper Small with quantization and CPU offloading may fit, but it is likely to be slow and memory-fragile on this GPU. The Base pilot is an engineering check, not the final accuracy benchmark.

Use a dedicated environment rather than the system Python. The training environment will require compatible versions of PyTorch with CUDA support, Transformers, Datasets, Accelerate, Evaluate, PEFT, audio-processing dependencies, and optionally bitsandbytes for quantized model loading.

Follow [`WHISPER_TRAINING_ENVIRONMENT_SETUP.md`](WHISPER_TRAINING_ENVIRONMENT_SETUP.md) for the tested Windows CUDA 12.6 installation and verification commands.

## 7. Fine-tune Large-v3 on a temporary GPU

For the production experiment, use a Linux machine with:

- One NVIDIA GPU with at least 24 GB VRAM
- 32-64 GB system memory
- Sufficient SSD space for the model, dataset, feature cache, and checkpoints

Use parameter-efficient LoRA training instead of updating all 1.55 billion parameters. Load the frozen base model in 8-bit when supported.

Starting configuration:

```yaml
model: openai/whisper-large-v3
load_in_8bit: true

lora_rank: 8
lora_alpha: 16
lora_dropout: 0.05
target_modules:
  - q_proj
  - v_proj

per_device_train_batch_size: 1
gradient_accumulation_steps: 16
gradient_checkpointing: true
mixed_precision: fp16

learning_rate: 0.0001
num_train_epochs: 3-5
predict_with_generate: true

language: Chinese
task: transcribe
```

Use BF16 instead of FP16 when the selected GPU supports it well. Treat the parameters above as a starting point rather than fixed values. At minimum, compare learning rates of `5e-5`, `1e-4`, and `2e-4`. Stop early when validation exact-identifier accuracy stops improving or the false-accept rate begins to degrade.

Save LoRA adapters separately during experimentation because they are much smaller than full model checkpoints. Preserve the base-model revision, dependency versions, dataset revision, random seed, and complete training configuration for every run.

## 8. Evaluate and select a checkpoint

For each candidate checkpoint:

1. Generate full transcripts for the locked test recordings.
2. Apply the same production identifier normalization and extraction logic.
3. Compare extracted identifiers with speech ground truth.
4. Run the complete speech-versus-OCR verification tests.
5. Inspect errors by speaker, speed, microphone, noise condition, identifier character, and sentence template.

Select the checkpoint using exact identifier performance and verification safety. A model with a slightly better general transcription score should not be selected if it produces more false matches.

Maintain a confusion report for individual characters and transitions. Add genuine production failures to a separate review pool, correct their labels, and include approved examples in a later training-data revision rather than silently modifying the locked test set.

## 9. Export for local inference

After selecting the adapter:

1. Load the original Large-v3 base model.
2. Apply and merge the selected LoRA adapter.
3. Save the merged Transformers model.
4. Convert it to CTranslate2 format.
5. Quantize the deployment model to INT8.
6. Test it through Faster-Whisper with batch size 1.

The T550's 4 GB VRAM makes Large-v3 INT8 inference borderline. Keep concurrency at one, avoid large batches, and benchmark beam sizes against both latency and memory use. If Large-v3 is unstable or too slow, compare it with a fine-tuned Whisper Small INT8 or FP16 model using exactly the same locked test set.

For this narrow identifier-recognition problem, a fine-tuned smaller model combined with strict extraction can be preferable to an unfine-tuned larger model. The deployment choice should be determined by measured exact-string accuracy, false-accept rate, latency, and memory stability.

## 10. Production safeguards

- Require a valid identifier on both the speech and OCR sides.
- Reject missing, malformed, or conflicting identifiers.
- Treat low-confidence or ambiguous cases as review-required.
- Log the raw transcript, extracted values, comparison result, model version, and OCR version for traceability.
- Monitor accuracy by operator and environment without using production data for retraining until it has been reviewed and labelled.
- Version the normalization and extraction rules together with the ASR model.
- Keep the OCR identifier out of the ASR decoding prompt to preserve independent verification.

## Suggested experiment sequence

1. Freeze the first representative test set and measure the current baseline.
2. Train Whisper Base LoRA locally on a small data subset to validate the complete workflow.
3. Train Whisper Small LoRA and measure whether the narrow-domain accuracy is already sufficient.
4. Train Large-v3 LoRA on a 24 GB GPU using the same data and metrics.
5. Compare Small and Large-v3 after deployment quantization.
6. Deploy only after mismatch tests demonstrate an acceptable false-accept rate.

This sequence avoids spending substantial compute before the data, labels, extraction rules, and safety metrics have been verified end to end.
