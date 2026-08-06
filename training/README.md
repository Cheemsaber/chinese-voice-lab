# Whisper Base LoRA Smoke Test

This directory contains a deliberately bounded engineering test for the local
Whisper fine-tuning pipeline. It verifies manifest loading, TorchCodec audio
decoding, Whisper preprocessing, LoRA attachment, two optimizer steps,
validation loss, adapter saving, and one validation transcription.

It is not an accuracy experiment. The script never selects a record whose
manifest split is `test`.

## Files

- `smoke_test.py`: validates the manifest and runs the Python training test.
- `run_smoke.ps1`: selects the project venv and supplies safe Windows defaults.

## Fixed safety boundary

The default run uses:

```text
Model:              openai/whisper-base
Training records:   4
Validation records: 1
Batch size:         1
Optimizer steps:    2
LoRA rank:           8
LoRA targets:        q_proj and v_proj
Test records:        0
```

Records are sorted by manifest `id` before selection, making the subset
deterministic. The script rejects malformed labels, missing paths, duplicate
IDs, and speaker or session leakage before loading a model.

## Prerequisites

The expected environment is:

```text
E:\chinese-voice-lab\.venv-whisper-ft
```

Install dependencies from the repository root if needed:

```powershell
& .\.venv-whisper-ft\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip check
```

The default Hugging Face cache used by the wrapper is:

```text
E:\huggingface-cache
```

## 1. Run the dataset-only preflight

This command does not load or download a model and does not create a training
checkpoint:

```powershell
cd E:\chinese-voice-lab
& .\training\run_smoke.ps1 -PreflightOnly
```

Expected final line:

```text
Dataset-only preflight: PASS
```

## 2. Cache the Transformers model

The existing Systran Faster-Whisper cache cannot be used for Transformers
training. Download the original model once:

```powershell
$env:HF_HOME = "E:\huggingface-cache"
$env:HF_HUB_DOWNLOAD_TIMEOUT = "120"
$env:HF_HUB_ETAG_TIMEOUT = "30"

& .\.venv-whisper-ft\Scripts\hf.exe download openai/whisper-base
```

Verify offline loading before training:

```powershell
$env:HF_HOME = "E:\huggingface-cache"
$env:HF_HUB_OFFLINE = "1"

& .\.venv-whisper-ft\Scripts\python.exe -c "from transformers import WhisperProcessor, WhisperForConditionalGeneration; WhisperProcessor.from_pretrained('openai/whisper-base', local_files_only=True); WhisperForConditionalGeneration.from_pretrained('openai/whisper-base', local_files_only=True); print('Offline model load: PASS')"
```

## 3. Run the bounded smoke test

The wrapper uses offline mode by default, ensuring the complete checkpoint is
already cached before GPU work starts:

```powershell
cd E:\chinese-voice-lab
& .\training\run_smoke.ps1
```

If you intentionally want the training command to download missing model files:

```powershell
& .\training\run_smoke.ps1 -AllowDownload
```

Separating download from training is preferred on an unreliable connection.

## Outputs

The default output is:

```text
E:\chinese-voice-lab\training_output\smoke-base\
├── checkpoint-2\
├── final_adapter\
└── smoke_summary.json
```

The summary records the exact train and validation IDs, loss values, validation
transcription, extracted device IDs, and adapter location.

The command refuses to reuse a nonempty output directory. Select another path,
or explicitly permit reuse:

```powershell
& .\training\run_smoke.ps1 `
    -OutputDir "E:\chinese-voice-lab\training_output\smoke-base-run2"
```

Use `-OverwriteOutput` only when replacing files in the named smoke directory
is intentional.

The wrapper defaults to `-Precision fp32`. This is deliberate for the current
NVIDIA T550: a real selected batch produces a finite loss in FP32 but a NaN loss
under FP16 autocast. Keep FP32 for the 4 GB laptop smoke test. On the RTX A6000,
use `-Precision bf16` after confirming `torch.cuda.is_bf16_supported()` returns
`True`:

```powershell
& .\training\run_smoke.ps1 -Precision bf16
```

`fp16` remains available as an explicit diagnostic option, but it is not the
safe default for this dataset/model/hardware combination.

## Successful completion

A successful run must show:

- selected training and validation records;
- CUDA-backed Whisper Base loading;
- a small LoRA trainable-parameter count;
- finite training and validation losses;
- two completed optimizer steps;
- a saved `final_adapter` directory;
- one generated validation transcript; and
- `Smoke test: PASS`.

Recognition of the validation identifier is reported but is not required for
the engineering smoke test to pass. Five validation records are not enough to
measure production accuracy.

## Troubleshooting

### Model is not cached

Run the cache command above. The smoke test uses `local_files_only=True` unless
`-AllowDownload` is supplied.

### CUDA out of memory

Close other GPU applications and rerun the same command. Do not increase batch
size on the 4 GB T550. The script already uses batch size 1 and gradient
checkpointing.

### Output directory is not empty

Use a new `-OutputDir`. This preserves evidence from previous runs.

### Network timeout

Rerun the separate `hf download` command. Hugging Face cache downloads are
resumable; do not delete the partial cache.
