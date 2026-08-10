param(
    [string]$DatasetRoot = "G:\speech-dataset",
    [string]$Model = "openai/whisper-base",
    [string]$OutputDir = "G:\chinese-voice-lab\training_output\smoke-base",
    [int]$TrainSamples = 4,
    [int]$ValidationSamples = 1,
    [int]$MaxSteps = 2,
    [ValidateSet("fp32", "fp16", "bf16")]
    [string]$Precision = "fp32",
    [switch]$PreflightOnly,
    [switch]$AllowDownload,
    [switch]$AllowCpu,
    [switch]$OverwriteOutput
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot ".venv-whisper-ft\Scripts\python.exe"
$script = Join-Path $PSScriptRoot "smoke_test.py"
$manifest = Join-Path $DatasetRoot "manifests\metadata.jsonl"

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Training Python not found: $python"
}
if (-not (Test-Path -LiteralPath $script -PathType Leaf)) {
    throw "Smoke-test entry point not found: $script"
}
if (-not (Test-Path -LiteralPath $manifest -PathType Leaf)) {
    throw "Dataset manifest not found: $manifest"
}

if ([string]::IsNullOrWhiteSpace($env:HF_HOME)) {
    $env:HF_HOME = "G:\huggingface-cache"
}
$env:HF_HUB_DOWNLOAD_TIMEOUT = "120"
$env:HF_HUB_ETAG_TIMEOUT = "30"
$env:TOKENIZERS_PARALLELISM = "false"
$env:PYTHONUTF8 = "1"
$OutputEncoding = [Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)

if ($AllowDownload) {
    Remove-Item Env:HF_HUB_OFFLINE -ErrorAction SilentlyContinue
} else {
    $env:HF_HUB_OFFLINE = "1"
}

$smokeArgs = @(
    $script,
    "--dataset-root", $DatasetRoot,
    "--manifest", $manifest,
    "--model", $Model,
    "--output-dir", $OutputDir,
    "--train-samples", $TrainSamples,
    "--validation-samples", $ValidationSamples,
    "--max-steps", $MaxSteps,
    "--precision", $Precision
)

if ($PreflightOnly) {
    $smokeArgs += "--preflight-only"
}
if ($AllowDownload) {
    $smokeArgs += "--allow-download"
}
if ($AllowCpu) {
    $smokeArgs += "--allow-cpu"
}
if ($OverwriteOutput) {
    $smokeArgs += "--overwrite-output"
}

Write-Host "Python:     $python"
Write-Host "Dataset:    $DatasetRoot"
Write-Host "Manifest:   $manifest"
Write-Host "Model:      $Model"
Write-Host "Precision:  $Precision"
Write-Host "HF_HOME:    $env:HF_HOME"
Write-Host "Offline:    $env:HF_HUB_OFFLINE"
Write-Host "Output:     $OutputDir"

& $python @smokeArgs
if ($LASTEXITCODE -ne 0) {
    throw "Smoke test failed with exit code $LASTEXITCODE"
}
