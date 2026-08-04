# Whisper Training Environment Setup

This guide installs the packages needed to validate the speech dataset and run
a small Whisper LoRA fine-tuning pilot on Windows. It is written for the
project environment at:

```text
E:\chinese-voice-lab\.venv-whisper-ft
```

The current tested foundation is Python 3.11 with the CUDA 12.6 builds of
PyTorch 2.13.0 and TorchVision 0.28.0.

## 1. Activate only the training environment

If PowerShell shows both `(whisper-ft)` and `(base)`, leave both environments
and then reactivate only the venv:

```powershell
deactivate
conda deactivate
cd E:\chinese-voice-lab
& .\.venv-whisper-ft\Scripts\Activate.ps1
```

If the prompt already shows only `(whisper-ft)`, do not deactivate it. Verify
the active interpreter instead:

```powershell
python --version
python -c "import sys; print(sys.executable)"
python -m pip --version
```

The Python and pip paths must both begin with:

```text
E:\chinese-voice-lab\.venv-whisper-ft
```

Do not continue if either command points to Anaconda, the inference venv, or a
system Python installation.

## 2. Upgrade the packaging tools

Run:

```powershell
python -m pip install --upgrade pip setuptools wheel
```

These are environment bootstrap tools, so they are intentionally not pinned as
project runtime dependencies in `requirements.txt`:

| Tool | Function |
| --- | --- |
| `pip` | Finds, resolves, downloads, installs, and uninstalls Python packages. |
| `setuptools` | Provides a common build backend and compatibility support for packages distributed from source. |
| `wheel` | Builds and understands `.whl` binary package archives, avoiding source compilation when a compatible wheel is available. |

The correct syntax is `python -m pip install`. `python -m install` is invalid
because Python has no standard module named `install`.

## 3. Install the complete project dependency set

From the repository root, run:

```powershell
cd E:\chinese-voice-lab
python -m pip install -r requirements.txt
```

This single command installs both the existing Faster-Whisper inference stack
and the new training stack. Already-installed compatible packages are retained.

The first line of `requirements.txt` adds the official CUDA 12.6 PyTorch wheel
index as an additional index. PyPI remains available for all non-PyTorch
packages. The `+cu126` pins ensure that pip selects the CUDA-enabled PyTorch
builds rather than CPU-only builds.

Do not replace `--extra-index-url` with a global PyTorch-only `--index-url` in
the requirements file. The PyTorch index does not contain the complete
Hugging Face and audio dependency set.

## 4. Functions of the new training packages

| Package | Function in this project |
| --- | --- |
| `torch` | Runs Whisper tensors, automatic differentiation, mixed-precision CUDA operations, and optimizer updates. |
| `torchvision` | Supplies PyTorch vision utilities. Whisper does not directly need it, but it was installed by the selected official PyTorch command and is pinned for environment reproducibility. |
| `transformers` | Provides the Whisper model, processor, tokenizer, generation code, training arguments, and sequence-to-sequence trainer. |
| `datasets[audio]` | Loads JSONL/local dataset records, manages train-validation-test splits, maps preprocessing functions, and enables audio-column decoding support. |
| `accelerate` | Connects Transformers training to CPU/GPU devices, mixed precision, gradient accumulation, and later multi-GPU execution. |
| `evaluate` | Supplies a standard interface for evaluation metrics used during validation and checkpoint comparison. |
| `jiwer` | Computes speech-recognition WER and CER. CER is useful for diagnosing individual Chinese, digit, and letter errors. Device-ID exact accuracy should remain the primary project metric. |
| `peft` | Implements LoRA adapters so only a small subset of parameters is trained, reducing VRAM and checkpoint size. |
| `soundfile` | Reads and writes WAV/FLAC audio through libsndfile and exposes samples as NumPy arrays. |
| `librosa` | Provides audio inspection, resampling, duration, and signal-processing utilities useful in dataset validation. |
| `torchcodec` | Decodes media into tensors for current Hugging Face audio dataset pipelines. |
| `tensorboard` | Records and visualizes training loss, validation metrics, learning rate, and other run statistics. |

`bitsandbytes` is not included in the initial Windows pilot. It is useful for
some 8-bit or 4-bit training workflows, but it adds another CUDA-sensitive
binary dependency. Add it only when the basic Whisper Base LoRA pipeline works
and a specific training script actually uses quantized loading.

## 5. Verify the installation

First check dependency consistency:

```powershell
python -m pip check
```

The expected result is:

```text
No broken requirements found.
```

Then verify imports:

```powershell
python -c "import torch, transformers, datasets, accelerate, evaluate, jiwer, peft, soundfile, librosa, torchcodec, tensorboard; print('Training imports: PASS')"
```

Verify CUDA and the selected GPU:

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA runtime:', torch.version.cuda); print('GPU available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

On the current laptop, the important results are:

```text
CUDA runtime: 12.6
GPU available: True
GPU: NVIDIA T550 Laptop GPU
```

Finally, confirm the key pinned versions:

```powershell
python -m pip show torch torchvision transformers datasets accelerate evaluate jiwer peft soundfile librosa torchcodec tensorboard
```

## 6. Why `tokenizers` is pinned to 0.22.2

The previous inference environment recorded `tokenizers==0.23.1`. Current
`transformers==5.14.1` requires a Tokenizers version from 0.22.0 through
0.23.0. Keeping 0.23.1 makes pip stop with `ResolutionImpossible`. PyPI does
not provide a final 0.23.0 release, so the requirements use the latest stable
compatible version rather than a 0.23.0 release candidate.

The shared requirements therefore use:

```text
tokenizers==0.22.2
```

This version satisfies Transformers and remains compatible with the existing
Faster-Whisper stack.

The shared `fsspec` dependency is pinned to 2026.4.0 because Datasets 5.0.0
does not accept newer releases. This remains within Hugging Face Hub's
supported range.

## 7. Reproducing the environment on the RTX A6000 laptop

Install a stable Python 3.11 interpreter on that laptop, clone or copy the
repository, create a fresh venv, and run the same requirements command:

```powershell
python -m venv --prompt="whisper-ft" .venv-whisper-ft
& .\.venv-whisper-ft\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip check
```

The CUDA 12.6 wheels can run on the A6000 when its NVIDIA driver is sufficiently
new. A higher CUDA wheel number is not required merely because the GPU is more
powerful. Change the PyTorch CUDA build only after checking that laptop's driver
and retesting the complete dependency set.

## 8. Maintaining dependency records

When adding a package that the project imports directly:

1. Install and test it inside `.venv-whisper-ft`.
2. Read its installed version with `python -m pip show PACKAGE`.
3. Add that exact version to `requirements.txt`.
4. Run `python -m pip check` and the import checks again.

Avoid blindly replacing `requirements.txt` with `pip freeze`. A freeze includes
every transitive package and would also lose the explanatory structure and the
explicit CUDA index configuration. If a complete environment snapshot is
needed for an experiment, save it separately:

```powershell
python -m pip freeze --all > requirements-lock.txt
```

Keep `requirements.txt` as the reviewed installation specification and use the
lock file only as a run-specific record.
