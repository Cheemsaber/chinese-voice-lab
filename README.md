# Chinese Voice Lab

Chinese Voice Lab is an experimental speech-recognition project for Chinese
operational utterances containing structured equipment identifiers such as
`019KG`. It covers the path from Windows microphone capture and controlled
Faster-Whisper benchmarks to reproducible Hugging Face Whisper LoRA training,
generation-based evaluation, and adapter export.

> **Project status:** This repository is public for technical review and
> reproducibility, but it is **not currently accepting external contributions**.
> Please do not submit pull requests, dataset additions, or feature requests at
> this stage.

> **Private data:** The training dataset is not included in this repository and
> is not available for public access. Consent to publish the participants' voice
> recordings has not been granted, so the training audio, manifests,
> participant metadata, and utterance-level outputs must remain private.

This is a research prototype, not a production- or safety-certified speech
system.

## Project goals

The project investigates four connected questions:

1. How can the first phoneme be captured reliably through Windows audio APIs?
2. How well do pretrained Whisper models recognize Chinese speech mixed with
   letter-and-number equipment identifiers?
3. Can parameter-efficient LoRA adaptation improve identifier and transcript
   accuracy on a small domain dataset?
4. How can training and evaluation remain reproducible while keeping private
   voice data outside the public repository?

## Achievements

- Isolated the original weak-onset problem primarily to the microphone input
  path rather than Faster-Whisper decoding. Realtek MME device 1, native
  44.1 kHz capture, early stream opening, and retained pre-roll produced the
  most reliable tested input.
- Demonstrated that arbitrary leading silence is not a reliable recognition
  fix and that beam size 5 added latency without improving the controlled
  sample. Beam size 1 was sufficient in that benchmark.
- Built a repeatable mixed-identifier benchmark. The recorded Whisper Small
  run recognized 14 of 16 equipment-identifier cases exactly and exposed the
  remaining domain errors that motivated fine-tuning.
- Built a bounded Whisper Base LoRA smoke test covering manifest validation,
  audio decoding, feature extraction, LoRA injection, two optimizer steps,
  validation, generation, and adapter saving. The recorded smoke run passed
  and recovered the expected `019KG` identifier.
- Built a configuration-driven full training pipeline with speaker/session
  split isolation, complete audio preflight, offline model loading, checkpoint
  resume, BF16 training, gradient checkpointing, early stopping, reproducibility
  metadata, prediction export, and optional held-out test evaluation.
- Completed LoRA experiments for Whisper Base, Small, and Large-v3, targeting
  `q_proj` and `v_proj` with ranks 8, 16, and 32 where applicable.
- Reduced Whisper Small validation CER from `0.3333` to `0.0622` with the best
  rank-32 adapter, an 81.34% relative reduction.
- Reduced Whisper Large-v3 CER by 7.76% relative on validation and 7.34%
  relative on the held-out test while preserving its already strong identifier
  accuracy.

## Repository layout

```text
.
|-- main.py                              # Early microphone/inference prototype
|-- experiments/                         # Controlled capture and ASR benchmarks
|-- experiment_output/                   # Small public experiment artifacts
|-- training/
|   |-- smoke_test.py                    # Bounded engineering smoke test
|   |-- run_smoke.ps1                    # Smoke-test launcher
|   |-- train_lora.py                    # Full Seq2SeqTrainer entry point
|   |-- lora_common.py                   # Data, model, metric, and artifact helpers
|   |-- run_train.ps1                    # Full-training launcher
|   |-- evaluate_lora.py                 # Frozen-baseline or adapter evaluation
|   `-- configs/                          # One YAML file per experiment
|-- requirements.txt                     # Reproducible Python dependency set
|-- FULL_DATA_LORA_TRAINING_GUIDE.md      # Detailed training guide
|-- WHISPER_FINETUNING_WORKFLOW.md        # Dataset-to-deployment workflow
`-- WHISPER_TRAINING_ENVIRONMENT_SETUP.md # Windows/CUDA environment setup
```

Generated checkpoints, adapters, predictions, and run metadata are written to
`training_output/`, which is intentionally ignored by Git.

## Dataset and privacy boundary

The current private manifest contains 142 recordings:

| Split | Recordings | Speakers | Sessions | Purpose |
| --- | ---: | ---: | ---: | --- |
| Train | 106 | 14 | 14 | Optimization |
| Validation | 16 | 2 | 3 | Model and checkpoint selection |
| Test | 20 | 2 | 3 | Held-out final evaluation |

Speaker and session identities are isolated across splits. The training
pipeline rejects duplicate IDs, missing files, malformed records, path escapes,
and cross-split speaker/session leakage before loading a model.

Audio under the private `processed_16khz` directory is normalized to mono,
16 kHz, 16-bit PCM WAV. This gives the training pipeline one stable input
format, avoids repeated resampling differences, and matches Whisper's expected
sampling rate before log-Mel feature extraction.

The private dataset must not be committed, mirrored, attached to releases, or
shared through issue reports. Generated prediction files can reproduce private
transcripts and should be handled with the same care.

The small WAV files already present under `experiment_output/` and the
repository root are separate from the training corpus. They should remain
public only if the repository owner has confirmed publication rights for every
recorded voice.

## Training pipeline

```text
private JSONL manifest
        |
        v
schema, path, speaker, and session validation
        |
        v
decode every WAV and verify 16 kHz audio
        |
        v
WhisperProcessor
  |-- feature extractor: waveform -> log-Mel input_features
  `-- tokenizer: transcript -> label token IDs
        |
        v
pretrained Whisper encoder-decoder
        +
PEFT LoRA adapters on q_proj and v_proj
        |
        v
Seq2SeqTrainer
  |-- teacher-forced forward pass and cross-entropy loss
  |-- PyTorch autograd backward pass
  |-- AdamW updates only trainable LoRA parameters
  `-- generation-based validation and checkpointing
        |
        v
best_adapter + processor + aggregate metrics
        |
        v
one final evaluation on the locked test split
```

For each selected attention projection, LoRA uses

```text
y = W x + (alpha / rank) B A x
```

The pretrained Whisper weight `W` remains frozen while the low-rank `A` and
`B` matrices are trained. Current experiments keep `alpha / rank = 2`, use
5% LoRA dropout, and target query and value projections.

## Current results

The tables below summarize locally recorded aggregate metrics. Lower CER is
better; higher identifier exact accuracy is better. Because the validation
split contains only 16 recordings, these values are experimental rather than
production estimates.

### Validation

| Model | Configuration | Identifier exact accuracy | CER |
| --- | --- | ---: | ---: |
| Whisper Base baseline | Frozen pretrained model | 0.3125 | 0.8483 |
| Whisper Base LoRA | r8, alpha 16, LR 1e-4, seed 42/44 tie | 0.7500 | 0.2960 |
| Whisper Small baseline | Frozen pretrained model | 0.6875 | 0.3333 |
| Whisper Small LoRA | r32, alpha 64, LR 5e-5, seed 42 | **0.8750** | **0.0622** |
| Whisper Large-v3 baseline | Frozen pretrained model | **0.9375** | 0.2886 |
| Whisper Large-v3 LoRA | r32, alpha 64, LR 5e-5, seed 42 | **0.9375** | **0.2662** |

### Held-out test

| Model | Configuration | Identifier exact accuracy | CER |
| --- | --- | ---: | ---: |
| Whisper Small LoRA | r32, alpha 64, LR 5e-5, seed 42 | 0.9500 | 0.0842 |
| Whisper Large-v3 baseline | Frozen pretrained model | 1.0000 | 0.2238 |
| Whisper Large-v3 LoRA | r32, alpha 64, LR 5e-5, seed 42 | 1.0000 | **0.2074** |

A saved Whisper Small baseline test artifact is not currently part of the
experiment record, so it is intentionally omitted from the held-out table.
WER is also recorded, but CER is the more useful general transcription metric
for these predominantly Chinese utterances because whitespace-based word
segmentation is not stable.

## Quick start

The project is designed for Windows PowerShell, Python 3.11, an NVIDIA CUDA
environment, and locally cached Hugging Face models. Follow the complete
[environment setup guide](WHISPER_TRAINING_ENVIRONMENT_SETUP.md) before running
GPU training.

```powershell
py -3.11 -m venv .venv-whisper-ft
& .\.venv-whisper-ft\Scripts\Activate.ps1
python -m pip install -r .\requirements.txt
python -m pip check
```

The committed YAML configurations contain machine-specific absolute paths for
the dataset, cache, and outputs. Review and change those paths before using the
project on another computer. Do not replace them with a path to public storage
containing the private corpus.

### Validate data without loading a model

```powershell
& .\training\run_smoke.ps1 `
    -DatasetRoot "D:\private-speech-dataset" `
    -PreflightOnly
```

### Run the bounded smoke test

```powershell
& .\training\run_smoke.ps1 `
    -DatasetRoot "D:\private-speech-dataset" `
    -Precision fp32
```

The smoke test defaults to 4 training records, 1 validation record, and 2
optimizer steps. It never uses the test split.

### Validate a full-training configuration

```powershell
& .\training\run_train.ps1 `
    -Config .\training\configs\small_lora_r32_lr5e5_seed42.yaml `
    -ValidateOnly
```

### Train an adapter

```powershell
& .\training\run_train.ps1 `
    -Config .\training\configs\small_lora_r32_lr5e5_seed42.yaml
```

Use `-ResumeFromCheckpoint` to resume a compatible interrupted run. Use
`-EvaluateTest` only for a final selected configuration; the test split should
not participate in routine tuning.

### Evaluate a frozen baseline or saved adapter

```powershell
# Frozen baseline: omit --adapter
& .\.venv-whisper-ft\Scripts\python.exe `
    .\training\evaluate_lora.py `
    --config .\training\configs\whisper_small_baseline.yaml `
    --split validation

# Adapter evaluation
& .\.venv-whisper-ft\Scripts\python.exe `
    .\training\evaluate_lora.py `
    --config .\training\configs\small_lora_r32_lr5e5_seed42.yaml `
    --adapter .\training_output\small-lora-r32-lr5e5-seed42\best_adapter `
    --split test
```

The launchers use offline Hugging Face mode by default. Supply the documented
download option only when intentionally populating a local model cache.

## Metrics

- **Identifier exact accuracy:** proportion of utterances whose complete set
  of identifiers matches the reference.
- **Identifier character accuracy:** character-level accuracy inside normalized
  identifiers.
- **Identifier precision/recall:** false-positive and missed-identifier
  behavior.
- **CER:** character edit distance divided by the number of reference
  characters.
- **WER:** whitespace-token word error rate; retained for comparability but
  interpreted cautiously for Chinese.

Predictions are normalized for Unicode, whitespace, case, and configured
Traditional-to-Simplified Chinese conversion before aggregate evaluation.

## Current limitations and known issues

### 1. Checkpoint selection uses an overly coarse metric

The current Trainer selects and early-stops on identifier exact accuracy only.
With 16 validation recordings, one utterance changes that metric by 6.25
percentage points, and tied scores retain the earliest checkpoint.

This affected the Large-v3 rank-32 run:

```text
Epoch 1: identifier accuracy 0.9375, CER 0.2662  <- selected
Epoch 4: identifier accuracy 0.9375, CER 0.0572  <- not selected
```

Validation loss and CER were still improving, so this is not evidence of
classic overfitting. It is a localized checkpoint-selection issue and should
be straightforward to fix by selecting lexicographically: highest identifier
accuracy first, then lowest CER among ties. Early stopping must monitor the
same rule.

### 2. The dataset is too small for higher-capacity experiments

The 106/16/20 split is useful for a controlled pilot but too small for strong
claims across speakers, microphones, accents, noise conditions, and operating
phrases. Rank 32 is therefore the current LoRA ceiling. Do **not** continue to
rank 64 or higher on this corpus: additional adapter capacity would add
parameters without enough independent data and would increase the risk of
memorization and unstable model selection.

The next scaling step should be more consented, speaker-diverse data and more
repeated seeds, not a higher LoRA rank.

### 3. Limited repeated-seed evidence

Several rank-8 runs use seeds 42-44, but the winning rank-32 Small and Large-v3
configurations currently have only seed 42. Repeat the final configurations
across multiple seeds and report mean and variation before treating rank 32 as
a stable improvement.

### 4. Validation-selection and test-reuse risk

Multiple configurations were compared on the same 16-record validation split.
That can overfit experiment choices to a small validation set even when the
training curve itself looks healthy. Small and Large-v3 have also now been
examined on the 20-record test split. Further tuning should not use those test
results as feedback; reserve a new consented audit set for the next final
comparison if development continues.

### 5. Domain coverage remains narrow

Current metrics emphasize identifiers matching `DDDCC`-style patterns and a
limited set of operational phrases. They do not yet establish robustness to
unseen equipment formats, spontaneous speech, overlapping speakers, heavy
noise, far-field microphones, or safety-critical semantic distinctions such
as negation, action, state, and numerical value.

### 6. Environment portability is limited

The training path currently assumes Windows, local absolute paths, CUDA/BF16,
a local Hugging Face cache, and a compatible Torch/TorchCodec/FFmpeg stack.
TorchCodec requires the appropriate shared FFmpeg libraries. The bounded smoke
test uses FP32 by default on the 4 GB NVIDIA T550 because FP16 produced a
non-finite loss in the recorded environment; full runs were designed around a
larger BF16-capable GPU.

### 7. Training and live inference are not integrated

`main.py` remains an early microphone prototype. It does not yet load the
selected PEFT adapter or implement a production streaming, confidence,
fallback, or human-verification workflow.

### 8. Public-release hygiene still needs review

The private training corpus is correctly external to the repository, but all
tracked experimental WAV files should be audited to confirm that each voice is
owned by the repository author or explicitly cleared for publication. The
repository also does not currently include a `LICENSE` file, so public
visibility should not be interpreted as a completed open-source release.

## Documentation

- [Full-dataset LoRA training guide](FULL_DATA_LORA_TRAINING_GUIDE.md)
- [Whisper fine-tuning workflow](WHISPER_FINETUNING_WORKFLOW.md)
- [Training environment setup](WHISPER_TRAINING_ENVIRONMENT_SETUP.md)
- [Smoke-test documentation](training/README.md)
- [Mixed-identifier benchmark](experiments/mixed_identifier_benchmark.py)
- [Recorded mixed-identifier results](experiment_output/mixed_identifier_benchmark/results_20260730_004523.csv)

## Contribution policy

External contributions are not being accepted at this time. Pull requests,
third-party voice datasets, and unsolicited code changes may be closed without
review. This policy protects the current experimental scope and prevents
private or insufficiently consented voice data from entering the project.

The policy may be reconsidered after the checkpoint-selection fix, dataset
governance review, publication-rights audit, and a formal license decision.
