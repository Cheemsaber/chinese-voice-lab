# Full-Dataset Whisper LoRA Training Guide

## Purpose

This guide describes how to add a separate, configuration-driven training path
for fine-tuning Whisper with LoRA on the existing dataset at:

```text
G:\speech-dataset
```

The design has three goals:

1. Switch Whisper models by changing one configuration value.
2. Modify LoRA settings without editing Python code.
3. Modify Hugging Face training arguments without editing Python code.

The existing bounded smoke test must remain intact. It is a pipeline diagnostic,
not the full training entry point.

## Current dataset and environment

The manifest currently contains 65 recordings:

| Split | Records | Duration | Speakers | Purpose |
|---|---:|---:|---:|---|
| `train` | 49 | 6.387 minutes | 6 | Gradient updates |
| `validation` | 8 | 1.305 minutes | 3 | Checkpoint selection |
| `test` | 8 | 0.951 minutes | 3 | Final locked evaluation |

Important: "use all existing data" means loading and evaluating every record in
its assigned split. It does **not** mean combining validation and test recordings
into the training split.

The tested environment is:

```text
Python:       3.11.9
PyTorch:      2.13.0+cu126
CUDA runtime: 12.6
GPU:          NVIDIA RTX A6000, 48 GB
BF16:         supported
FFmpeg:       8.1.1 full-shared
```

## Proposed file layout

Add a production training path alongside the smoke test:

```text
training/
├── smoke_test.py                     # Existing; keep bounded and stable
├── run_smoke.ps1                     # Existing; keep bounded and stable
├── train_lora.py                     # New configuration-driven trainer
├── run_train.ps1                     # New PowerShell wrapper
├── evaluate_lora.py                  # New baseline/final evaluation entry point
└── configs/
    ├── base_lora.yaml                # First full-data experiment
    ├── base_lora_lr5e5.yaml          # Learning-rate comparison
    ├── small_lora.yaml               # Optional later comparison
    └── large_v3_lora.yaml            # Optional later comparison
```

Do not turn `smoke_test.py` into the production trainer. The smoke test should
continue to select four training records, one validation record, and two steps.

## Step 1: Create the first training configuration

Create `training/configs/base_lora.yaml` with the following structure. Use
single quotes around Windows paths so YAML does not treat backslashes as escape
characters.

```yaml
run:
  name: 'base-lora-r8-lr1e4-seed42'
  output_dir: 'G:\chinese-voice-lab\training_output\base-lora-r8-lr1e4-seed42'
  overwrite_output: false
  resume_from_checkpoint: null

data:
  dataset_root: 'G:\speech-dataset'
  manifest: 'G:\speech-dataset\manifests\metadata.jsonl'
  train_split: 'train'
  validation_split: 'validation'
  test_split: 'test'
  audio_field: 'audio'
  text_field: 'text'
  sampling_rate: 16000

model:
  id: 'openai/whisper-base'
  language: 'Chinese'
  language_token: 'zh'
  task: 'transcribe'
  local_files_only: true
  dtype: 'bf16'
  use_safetensors: true
  low_cpu_mem_usage: true
  attn_implementation: 'sdpa'

lora:
  task_type: 'SEQ_2_SEQ_LM'
  r: 8
  lora_alpha: 16
  lora_dropout: 0.05
  target_modules:
    - 'q_proj'
    - 'v_proj'
  bias: 'none'

training:
  per_device_train_batch_size: 4
  per_device_eval_batch_size: 1
  gradient_accumulation_steps: 2
  learning_rate: 0.0001
  lr_scheduler_type: 'linear'
  warmup_ratio: 0.10
  num_train_epochs: 10
  bf16: true
  fp16: false
  gradient_checkpointing: true
  gradient_checkpointing_kwargs:
    use_reentrant: false
  eval_strategy: 'epoch'
  save_strategy: 'epoch'
  logging_strategy: 'steps'
  logging_steps: 1
  save_total_limit: 3
  predict_with_generate: true
  generation_max_length: 128
  load_best_model_at_end: true
  metric_for_best_model: 'identifier_exact_accuracy'
  greater_is_better: true
  remove_unused_columns: false
  dataloader_num_workers: 0
  optim: 'adamw_torch'
  report_to:
    - 'tensorboard'
  seed: 42
  data_seed: 42

early_stopping:
  enabled: true
  patience: 3
  threshold: 0.0

evaluation:
  identifier_pattern: '(?<!\d)\d{3}[A-Z]{2}'
  normalize_uppercase: true
  deduplicate_identifiers: true
  save_predictions: true
```

### Configuration rules

- `data.text_field` must remain `text`. It contains the full transcript.
- Do not train against `target_text`; it contains only the target identifier.
- `run.output_dir` must be unique for every experiment.
- Refuse to start when a non-empty output directory exists unless
  `overwrite_output` is explicitly true or a checkpoint is being resumed.
- Never load the test split into `Seq2SeqTrainer` during training.
- Copy the resolved configuration into the output directory before training.

## Step 2: Make model switching configuration-only

The trainer should read `model.id` and pass it to both the processor and model:

```python
processor = WhisperProcessor.from_pretrained(
    config["model"]["id"],
    language=config["model"]["language"],
    task=config["model"]["task"],
    local_files_only=config["model"]["local_files_only"],
)

model = WhisperForConditionalGeneration.from_pretrained(
    config["model"]["id"],
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=config["model"]["low_cpu_mem_usage"],
    use_safetensors=config["model"]["use_safetensors"],
    attn_implementation=config["model"]["attn_implementation"],
    local_files_only=config["model"]["local_files_only"],
)
```

The processor automatically loads the correct feature extractor. This matters
because Whisper Large-v3 uses 128 Mel bins while earlier Whisper models use 80.
Do not hard-code the number of Mel bins.

Use configuration profiles rather than editing code:

| Model | `model.id` | First micro-batch | Notes |
|---|---|---:|---|
| Base | `openai/whisper-base` | 4 | First full-data experiment |
| Small | `openai/whisper-small` | 2 | Try only if Base is insufficient |
| Large-v3 | `openai/whisper-large-v3` | 1 | Use BF16 and gradient accumulation |

When switching models, also change `run.name`, `run.output_dir`, and possibly
the micro-batch/accumulation values. Keep the effective batch size comparable.

Examples:

```yaml
# Base: effective batch 8
per_device_train_batch_size: 4
gradient_accumulation_steps: 2

# Small: effective batch 8
per_device_train_batch_size: 2
gradient_accumulation_steps: 4

# Large-v3: effective batch 8
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
```

## Step 3: Make LoRA configuration-only

The trainer should construct `LoraConfig` directly from the `lora` mapping:

```python
from peft import LoraConfig, TaskType, get_peft_model

task_types = {
    "SEQ_2_SEQ_LM": TaskType.SEQ_2_SEQ_LM,
}

lora_values = config["lora"]
lora_config = LoraConfig(
    task_type=task_types[lora_values["task_type"]],
    r=lora_values["r"],
    lora_alpha=lora_values["lora_alpha"],
    lora_dropout=lora_values["lora_dropout"],
    target_modules=lora_values["target_modules"],
    bias=lora_values["bias"],
)

model = get_peft_model(model, lora_config)
model.enable_input_require_grads()
model.print_trainable_parameters()
```

Recommended first values:

```yaml
r: 8
lora_alpha: 16
lora_dropout: 0.05
target_modules: ['q_proj', 'v_proj']
bias: 'none'
```

Change one LoRA variable at a time. Examples:

```yaml
# More adapter capacity, with more overfitting risk
r: 16
lora_alpha: 32

# Stronger regularization
lora_dropout: 0.10

# Broader attention adaptation; use only as a later experiment
target_modules: ['q_proj', 'k_proj', 'v_proj', 'out_proj']
```

Always save the exact LoRA configuration with the adapter. An adapter cannot be
interpreted correctly without knowing its base model and LoRA structure.

## Step 4: Make training arguments configuration-only

The trainer should validate the `training` mapping and then pass it into
`Seq2SeqTrainingArguments`:

```python
training_values = dict(config["training"])

training_args = Seq2SeqTrainingArguments(
    output_dir=config["run"]["output_dir"],
    run_name=config["run"]["name"],
    **training_values,
)
```

Do not blindly accept arbitrary YAML keys. Maintain an allowlist and fail on
unknown keys so misspellings do not silently produce unintended runs.

Recommended allowlist:

```python
ALLOWED_TRAINING_KEYS = {
    "per_device_train_batch_size",
    "per_device_eval_batch_size",
    "gradient_accumulation_steps",
    "learning_rate",
    "lr_scheduler_type",
    "warmup_ratio",
    "num_train_epochs",
    "bf16",
    "fp16",
    "gradient_checkpointing",
    "gradient_checkpointing_kwargs",
    "eval_strategy",
    "save_strategy",
    "logging_strategy",
    "logging_steps",
    "save_total_limit",
    "predict_with_generate",
    "generation_max_length",
    "load_best_model_at_end",
    "metric_for_best_model",
    "greater_is_better",
    "remove_unused_columns",
    "dataloader_num_workers",
    "optim",
    "report_to",
    "seed",
    "data_seed",
}
```

Validate these relationships before constructing the Trainer:

- Exactly one of `bf16` and `fp16` may be true.
- `bf16` requires `torch.cuda.is_bf16_supported()`.
- `gradient_accumulation_steps` must be at least one.
- `learning_rate` must be positive.
- `warmup_ratio` must be between zero and one.
- `eval_strategy` and `save_strategy` should match when
  `load_best_model_at_end` is true.
- With step-based strategies, `save_steps` must be a multiple of `eval_steps`.
- `metric_for_best_model` must be returned by `compute_metrics`.

## Step 5: Load every record while respecting its split

Reuse the validated manifest functions from `smoke_test.py`, but do not reuse
the bounded `select_records` call.

```python
records = load_and_validate_manifest(manifest, dataset_root)
assert_split_isolation(records)

train_records = sorted(
    (r for r in records if r["split"] == config["data"]["train_split"]),
    key=lambda r: r["id"],
)
validation_records = sorted(
    (r for r in records if r["split"] == config["data"]["validation_split"]),
    key=lambda r: r["id"],
)
test_records = sorted(
    (r for r in records if r["split"] == config["data"]["test_split"]),
    key=lambda r: r["id"],
)
```

Before loading a model, print and validate:

```text
train:      49
validation: 8
test:       8 (locked; not passed to Trainer)
```

Fail if any expected split is empty or if a speaker/session leaks between
splits.

Use the same audio preparation and collator behavior that passed the smoke
test:

- Decode with `datasets.Audio(sampling_rate=16000)` and TorchCodec.
- Validate finite, non-empty waveforms.
- Create input features with the selected model's processor.
- Tokenize the full `text` transcript.
- Pad labels and replace non-label padding positions with `-100`.

## Step 6: Generate full validation metrics

The smoke test evaluates loss and generates one example. Full training must
generate predictions for all eight validation records on every evaluation.

Provide `compute_metrics` to `Seq2SeqTrainer` and return at least:

```text
identifier_exact_accuracy
identifier_character_accuracy
identifier_recall
identifier_precision
wer
cer
```

Metric procedure:

1. Replace `-100` label positions with the tokenizer padding token.
2. Decode predictions and references with `skip_special_tokens=True`.
3. Extract identifiers from both decoded strings using the configured regex.
4. Normalize to uppercase and deduplicate repeated identifiers.
5. Compare the complete identifier lists or sets.
6. Calculate WER/CER over the full transcripts.

For each validation record, save:

```json
{
  "id": "record-id",
  "reference": "full reference transcript",
  "prediction": "full generated transcript",
  "expected_device_ids": ["019KG"],
  "predicted_device_ids": ["019KG"],
  "identifier_exact": true
}
```

The validation set has only eight recordings, so exact accuracy changes in
12.5-point increments. Use exact identifier accuracy as the primary metric, but
also inspect character accuracy, WER/CER, loss, and individual predictions.

## Step 7: Add best-checkpoint and early-stopping behavior

Pass an early stopping callback only when enabled:

```python
callbacks = []
if config["early_stopping"]["enabled"]:
    callbacks.append(
        EarlyStoppingCallback(
            early_stopping_patience=config["early_stopping"]["patience"],
            early_stopping_threshold=config["early_stopping"]["threshold"],
        )
    )
```

Configure Trainer with:

```python
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=prepared["train"],
    eval_dataset=prepared["validation"],
    data_collator=collator,
    processing_class=processor,
    compute_metrics=compute_metrics,
    callbacks=callbacks,
)
```

Because the validation set is tiny, retain several checkpoints and review ties
manually. Do not select a checkpoint using test-set results.

## Step 8: Create the PowerShell wrapper

`training/run_train.ps1` should only locate the environment, configure cache
behavior, and invoke the Python trainer.

Recommended interface:

```powershell
param(
    [Parameter(Mandatory = $true)]
    [string]$Config,
    [switch]$AllowDownload,
    [string]$ResumeFromCheckpoint
)
```

Required behavior:

```powershell
$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot ".venv-whisper-ft\Scripts\python.exe"
$trainer = Join-Path $PSScriptRoot "train_lora.py"

if ([string]::IsNullOrWhiteSpace($env:HF_HOME)) {
    $env:HF_HOME = "G:\huggingface-cache"
}

$env:TOKENIZERS_PARALLELISM = "false"
$env:PYTHONUTF8 = "1"
$OutputEncoding = [Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)

if ($AllowDownload) {
    Remove-Item Env:HF_HUB_OFFLINE -ErrorAction SilentlyContinue
} else {
    $env:HF_HUB_OFFLINE = "1"
}
```

The wrapper should print the resolved Python, configuration file, model ID,
dataset root, cache directory, output directory, precision, and offline status
before starting.

## Step 9: Cache a model before offline training

Whisper Base is already cached. For another model:

```powershell
cd G:\chinese-voice-lab
$env:HF_HOME = "G:\huggingface-cache"
& .\.venv-whisper-ft\Scripts\hf.exe download openai/whisper-small
```

For Large-v3:

```powershell
& .\.venv-whisper-ft\Scripts\hf.exe download openai/whisper-large-v3
```

If the network requires the local proxy:

```powershell
$env:HTTP_PROXY = "http://127.0.0.1:7890"
$env:HTTPS_PROXY = "http://127.0.0.1:7890"
```

After caching, keep normal training offline for reproducibility.

## Step 10: Establish the baseline before training

Use `evaluate_lora.py` without an adapter to evaluate the unmodified base model.
Save validation and locked-test predictions separately:

```text
training_output/baselines/whisper-base/validation_predictions.jsonl
training_output/baselines/whisper-base/test_predictions.jsonl
training_output/baselines/whisper-base/metrics.json
```

Do not alter hyperparameters in response to test-set results. Use validation for
all training decisions.

## Step 11: Run the first full Base experiment

After implementing and validating the new trainer:

```powershell
cd G:\chinese-voice-lab
& .\training\run_train.ps1 `
    -Config ".\training\configs\base_lora.yaml"
```

The first full run should use:

```text
Model:                 openai/whisper-base
Train/validation:      49 / 8
Precision:             BF16
Micro-batch:           4
Gradient accumulation: 2
Effective batch:       8
Learning rate:         1e-4
Maximum epochs:        10
Warmup:                10%
LoRA:                  r=8, alpha=16, dropout=0.05, q/v projections
```

With 49 training records and effective batch size eight, there are about seven
optimizer steps per epoch and at most about 70 steps over ten epochs.

## Step 12: Compare configurations correctly

Duplicate the YAML file and change one independent variable at a time.

Learning-rate comparison:

```text
base-lora-r8-lr5e5-seed42: learning_rate=0.00005
base-lora-r8-lr1e4-seed42: learning_rate=0.00010
```

Seed comparison:

```text
base-lora-r8-lr1e4-seed42
base-lora-r8-lr1e4-seed43
base-lora-r8-lr1e4-seed44
```

Model comparison:

```text
openai/whisper-base
openai/whisper-small
openai/whisper-large-v3
```

Keep the following unchanged when comparing models:

- Dataset revision and splits
- Transcript normalization
- Identifier extraction
- Effective batch size where practical
- Evaluation decoding settings
- Selection metrics

## Step 13: Evaluate the selected adapter

After selecting the best checkpoint using validation only:

```powershell
& .\.venv-whisper-ft\Scripts\python.exe `
    .\training\evaluate_lora.py `
    --config .\training\configs\base_lora.yaml `
    --adapter G:\chinese-voice-lab\training_output\base-lora-r8-lr1e4-seed42\best_adapter `
    --split test
```

Save the raw transcript and identifier result for every record. Compare the
fine-tuned test results with the frozen zero-shot baseline.

## Required output artifacts

Each run directory should contain:

```text
resolved_config.yaml
run_metadata.json
manifest.sha256
requirements_snapshot.txt
checkpoints/
best_adapter/
validation_predictions.jsonl
validation_metrics.json
train_metrics.json
trainer_state.json
tensorboard/
```

`run_metadata.json` should include:

```text
Git commit and dirty-worktree status
Base model ID and cached revision
Python version
PyTorch and CUDA runtime versions
GPU name and VRAM
LoRA configuration
Training arguments
Random seeds
Train and validation record IDs
Start/end timestamps
Peak allocated and reserved CUDA memory
```

## Resume behavior

Never resume into a different configuration silently. Before resuming, compare
the saved resolved configuration with the requested configuration.

Resume example:

```powershell
& .\training\run_train.ps1 `
    -Config ".\training\configs\base_lora.yaml" `
    -ResumeFromCheckpoint "G:\chinese-voice-lab\training_output\base-lora-r8-lr1e4-seed42\checkpoint-35"
```

The trainer should call:

```python
trainer.train(resume_from_checkpoint=checkpoint_path)
```

## Safety and quality gates

Do not advance to a larger model until the current model has passed these gates:

1. All training and validation audio decodes successfully.
2. Loss remains finite.
3. Adapter reload produces identical validation predictions.
4. Validation exact-identifier accuracy improves over zero-shot.
5. Character accuracy does not regress materially.
6. Full transcript quality remains acceptable.
7. No test records were used for training or checkpoint selection.
8. Results are reproducible across at least two seeds.

If Base meets the application requirement, retain Base. Try Small next only if
Base is insufficient. Use Large-v3 only after smaller models demonstrably fail.

## Data limitations to record with every result

The current dataset is appropriate for engineering experiments but too small for
a strong production claim:

- Only 49 training recordings and 6.387 minutes of training audio.
- Only eight validation and eight test recordings.
- Validation accuracy changes in increments of 12.5 percentage points.
- Every current recording contains at least one identifier, so false identifier
  detection on no-identifier speech cannot be measured adequately.

Before deployment, expand validation and test data with no-identifier commands,
malformed identifiers, conflicting identifiers, difficult speakers, new rooms,
new microphones, and realistic background noise.

## Final checklist

- [ ] Preserve the existing smoke test unchanged.
- [ ] Create one YAML configuration per experiment.
- [ ] Implement strict configuration validation.
- [ ] Load all 49 train and 8 validation records.
- [ ] Keep all 8 test records out of Trainer.
- [ ] Use full transcripts from `text` as labels.
- [ ] Generate predictions over the entire validation split.
- [ ] Save exact identifier, character, WER, and CER metrics.
- [ ] Cache the selected base model before offline training.
- [ ] Establish and save a zero-shot baseline.
- [ ] Run Base LoRA before attempting larger models.
- [ ] Select checkpoints using validation only.
- [ ] Evaluate the locked test set only after configuration selection.
- [ ] Save config, versions, revisions, record IDs, and predictions for every run.

## References

- Whisper Large-v3 model card: <https://huggingface.co/openai/whisper-large-v3>
- PEFT LoRA reference: <https://huggingface.co/docs/peft/main/package_reference/lora>
- Transformers Trainer reference: <https://huggingface.co/docs/transformers/main_classes/trainer>
