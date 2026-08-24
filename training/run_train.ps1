[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Config,
    [switch]$AllowDownload,
    [string]$ResumeFromCheckpoint,
    [switch]$OverwriteOutput,
    [switch]$ValidateOnly,
    [switch]$EvaluateTest
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot ".venv-whisper-ft\Scripts\python.exe"
$trainer = Join-Path $PSScriptRoot "train_lora.py"

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Project Python was not found: $python"
}
if (-not (Test-Path -LiteralPath $trainer -PathType Leaf)) {
    throw "Training entry point was not found: $trainer"
}
$configPath = (Resolve-Path -LiteralPath $Config).Path

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

$trainingArguments = @(
    $trainer,
    "--config", $configPath
)
if ($AllowDownload) {
    $trainingArguments += "--allow-download"
}
if (-not [string]::IsNullOrWhiteSpace($ResumeFromCheckpoint)) {
    $checkpointPath = (Resolve-Path -LiteralPath $ResumeFromCheckpoint).Path
    $trainingArguments += @("--resume-from-checkpoint", $checkpointPath)
}
if ($OverwriteOutput) {
    $trainingArguments += "--overwrite-output"
}
if ($ValidateOnly) {
    $trainingArguments += "--validate-only"
}
if ($EvaluateTest) {
    $trainingArguments += "--evaluate-test"
}

Write-Host "Python:      $python"
Write-Host "Config:      $configPath"
Write-Host "HF_HOME:     $env:HF_HOME"
Write-Host "Offline:     $env:HF_HUB_OFFLINE"
Write-Host "Resume:      $ResumeFromCheckpoint"
Write-Host "Validate:    $ValidateOnly"
Write-Host "Test:        $EvaluateTest"

& $python @trainingArguments
if ($LASTEXITCODE -ne 0) {
    throw "LoRA training failed with exit code $LASTEXITCODE"
}
