"""Run a bounded Whisper Base LoRA smoke test on the local speech dataset.

This is an engineering test for the dataset and training pipeline. It is not
an accuracy experiment. The script intentionally never loads the test split.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Audio, Dataset, DatasetDict
from peft import LoraConfig, get_peft_model
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    set_seed,
)
from transformers.utils import logging as transformers_logging


DEVICE_ID_PATTERN = re.compile(r"^\d{3}[A-Z]{2}$")
DEVICE_ID_SEARCH_PATTERN = re.compile(
    r"(?<!\d)\d{3}[A-Z]{2}"
)
REQUIRED_FIELDS = {
    "id",
    "audio",
    "raw_audio",
    "text",
    "target_text",
    "device_ids",
    "speaker_id",
    "session_id",
    "split",
    "is_known_device",
}
ALLOWED_SPLITS = {"train", "validation", "test"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the dataset and run a two-step Whisper Base LoRA smoke test."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(r"G:\speech-dataset"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(r"G:\speech-dataset\manifests\metadata.jsonl"),
    )
    parser.add_argument("--model", default="openai/whisper-base")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"G:\chinese-voice-lab\training_output\smoke-base"),
    )
    parser.add_argument("--train-samples", type=int, default=4)
    parser.add_argument("--validation-samples", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--precision",
        choices=("fp32", "fp16", "bf16"),
        default="fp32",
        help=(
            "Training precision. FP32 is the safe default for the 4 GB T550; "
            "use BF16 on a supported Ampere-or-newer training GPU."
        ),
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow Transformers to contact the Hub. The default requires a cached model.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate the manifest and decode selected audio without loading a model.",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow training without CUDA. Intended only for debugging.",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Permit reuse of a nonempty output directory.",
    )
    args = parser.parse_args()

    for name in ("train_samples", "validation_samples", "max_steps"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1")
    return args


def resolve_dataset_path(dataset_root: Path, value: str, field: str, record_id: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = dataset_root / candidate
    candidate = candidate.resolve()
    if not candidate.is_relative_to(dataset_root):
        raise ValueError(f"{record_id}: {field} escapes dataset root: {candidate}")
    if not candidate.is_file():
        raise FileNotFoundError(f"{record_id}: missing {field}: {candidate}")
    return candidate


def load_and_validate_manifest(manifest: Path, dataset_root: Path) -> list[dict[str, Any]]:
    dataset_root = dataset_root.resolve()
    manifest = manifest.resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    if not manifest.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {manifest}")

    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on manifest line {line_number}: {exc}") from exc

        missing = sorted(REQUIRED_FIELDS - set(record))
        if missing:
            raise ValueError(f"Manifest line {line_number} is missing fields: {missing}")

        record_id = str(record["id"]).strip()
        if not record_id:
            raise ValueError(f"Manifest line {line_number} has a blank id")
        if record_id in seen_ids:
            raise ValueError(f"Duplicate manifest id: {record_id}")
        seen_ids.add(record_id)

        split = record["split"]
        if split not in ALLOWED_SPLITS:
            raise ValueError(f"{record_id}: unsupported split {split!r}")
        if not isinstance(record["text"], str) or not record["text"].strip():
            raise ValueError(f"{record_id}: text must be nonempty")
        if not isinstance(record["device_ids"], list) or not record["device_ids"]:
            raise ValueError(f"{record_id}: device_ids must be a nonempty list")

        device_ids = [str(value) for value in record["device_ids"]]
        for device_id in device_ids:
            if not DEVICE_ID_PATTERN.fullmatch(device_id):
                raise ValueError(f"{record_id}: invalid device id {device_id!r}")
            if device_id not in record["text"]:
                raise ValueError(f"{record_id}: {device_id} is absent from text")
        if record["target_text"] != " ".join(device_ids):
            raise ValueError(f"{record_id}: target_text does not match device_ids")

        normalized = dict(record)
        normalized["id"] = record_id
        normalized["device_ids"] = device_ids
        normalized["audio"] = str(
            resolve_dataset_path(dataset_root, record["audio"], "audio", record_id)
        )
        normalized["raw_audio"] = str(
            resolve_dataset_path(dataset_root, record["raw_audio"], "raw_audio", record_id)
        )
        records.append(normalized)

    if not records:
        raise ValueError("Manifest contains no records")
    return records


def assert_split_isolation(records: list[dict[str, Any]]) -> None:
    for field in ("speaker_id", "session_id"):
        membership: dict[str, set[str]] = {}
        for record in records:
            membership.setdefault(str(record[field]), set()).add(record["split"])
        leaking = {key: sorted(value) for key, value in membership.items() if len(value) > 1}
        if leaking:
            raise ValueError(f"{field} leakage across splits: {leaking}")


def select_records(
    records: list[dict[str, Any]], split: str, count: int
) -> list[dict[str, Any]]:
    available = sorted(
        (record for record in records if record["split"] == split),
        key=lambda record: record["id"],
    )
    if len(available) < count:
        raise ValueError(f"Requested {count} {split} records, but only {len(available)} exist")
    return available[:count]


def build_dataset(
    train_records: list[dict[str, Any]], validation_records: list[dict[str, Any]]
) -> DatasetDict:
    return DatasetDict(
        {
            "train": Dataset.from_list(train_records).cast_column(
                "audio", Audio(sampling_rate=16_000)
            ),
            "validation": Dataset.from_list(validation_records).cast_column(
                "audio", Audio(sampling_rate=16_000)
            ),
        }
    )


def decode_audio(audio_decoder: Any) -> tuple[torch.Tensor, int]:
    samples = audio_decoder.get_all_samples()
    waveform = samples.data
    if waveform.ndim != 2:
        raise ValueError(f"Expected [channels, frames] audio, got {tuple(waveform.shape)}")
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0)
    else:
        waveform = waveform.squeeze(0)
    if waveform.numel() == 0 or not torch.isfinite(waveform).all():
        raise ValueError("Decoded audio is empty or contains non-finite samples")
    return waveform, int(samples.sample_rate)


def run_audio_preflight(dataset: DatasetDict) -> None:
    print("Selected dataset records:")
    for split in ("train", "validation"):
        print(f"  {split}: {len(dataset[split])}")
        for record in dataset[split]:
            waveform, sample_rate = decode_audio(record["audio"])
            if sample_rate != 16_000:
                raise ValueError(f"{record['id']}: expected 16000 Hz, got {sample_rate}")
            duration = waveform.numel() / sample_rate
            print(f"    {record['id']}: {duration:.3f}s, {record['text']}")


def prepare_dataset(dataset: DatasetDict, processor: WhisperProcessor) -> DatasetDict:
    def prepare_record(record: dict[str, Any]) -> dict[str, Any]:
        waveform, sample_rate = decode_audio(record["audio"])
        input_features = processor.feature_extractor(
            waveform.numpy(), sampling_rate=sample_rate
        ).input_features[0]
        labels = processor.tokenizer(record["text"]).input_ids
        return {"input_features": input_features, "labels": labels}

    prepared = DatasetDict()
    for split in ("train", "validation"):
        prepared[split] = dataset[split].map(
            prepare_record,
            remove_columns=dataset[split].column_names,
            keep_in_memory=True,
            load_from_cache_file=False,
            desc=f"Preparing {split} records",
        )
    return prepared


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor
    decoder_start_token_id: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1), -100
        )
        if (labels[:, 0] == self.decoder_start_token_id).all().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def generate_validation_example(
    model: torch.nn.Module,
    processor: WhisperProcessor,
    validation_record: dict[str, Any],
) -> dict[str, Any]:
    waveform, sample_rate = decode_audio(validation_record["audio"])
    inputs = processor.feature_extractor(
        waveform.numpy(), sampling_rate=sample_rate, return_tensors="pt"
    )
    device = next(model.parameters()).device
    model.eval()
    previous_transformers_verbosity = transformers_logging.get_verbosity()
    try:
        # Transformers 5.14 emits duplicate-logits-processor notices from the
        # Whisper generation wrapper. They are harmless and local to generation.
        transformers_logging.set_verbosity_error()
        with torch.inference_mode():
            generated_ids = model.generate(
                input_features=inputs.input_features.to(device),
                max_length=64,
                language="zh",
                task="transcribe",
            )
    finally:
        transformers_logging.set_verbosity(previous_transformers_verbosity)
    prediction = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    predicted_ids = DEVICE_ID_SEARCH_PATTERN.findall(prediction.upper())
    expected_ids = validation_record["device_ids"]
    result = {
        "record_id": validation_record["id"],
        "reference": validation_record["text"],
        "prediction": prediction,
        "expected_device_ids": expected_ids,
        "predicted_device_ids": predicted_ids,
        "device_ids_exact": predicted_ids == expected_ids,
    }
    print("Validation generation:")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def ensure_output_directory(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. "
            "Choose another directory or pass --overwrite-output."
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    records = load_and_validate_manifest(args.manifest, args.dataset_root)
    assert_split_isolation(records)
    train_records = select_records(records, "train", args.train_samples)
    validation_records = select_records(
        records, "validation", args.validation_samples
    )
    selected_ids = {record["id"] for record in train_records + validation_records}
    if any(record["split"] == "test" and record["id"] in selected_ids for record in records):
        raise AssertionError("A test record was selected")

    dataset = build_dataset(train_records, validation_records)
    run_audio_preflight(dataset)
    if args.preflight_only:
        print("Dataset-only preflight: PASS")
        return 0

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA is unavailable. Use --allow-cpu only for debugging.")
    if args.precision != "fp32" and not torch.cuda.is_available():
        raise RuntimeError(f"{args.precision.upper()} requires CUDA for this smoke test")
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 was requested, but this GPU does not support BF16")

    ensure_output_directory(args.output_dir, args.overwrite_output)
    local_files_only = not args.allow_download
    print(
        f"Loading {args.model} (local_files_only={local_files_only}) from "
        f"HF_HOME={os.environ.get('HF_HOME', Path.home() / '.cache' / 'huggingface')}"
    )
    processor = WhisperProcessor.from_pretrained(
        args.model,
        language="Chinese",
        task="transcribe",
        local_files_only=local_files_only,
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model,
        local_files_only=local_files_only,
    )
    model.config.use_cache = False
    model.generation_config.language = "zh"
    model.generation_config.task = "transcribe"

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.print_trainable_parameters()

    # In offline mode PEFT cannot query the Hub to re-confirm the base config.
    # This run never resizes the tokenizer, so the vocabulary warning is noise.
    warnings.filterwarnings(
        "ignore",
        message=r"Could not find a config file in .* - will assume that the vocabulary was not modified\.",
        module=r"peft\.utils\.save_and_load",
    )

    prepared = prepare_dataset(dataset, processor)
    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )
    use_fp16 = args.precision == "fp16"
    use_bf16 = args.precision == "bf16"
    print(f"Training precision: {args.precision.upper()}")
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        fp16=use_fp16,
        bf16=use_bf16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps",
        eval_steps=1,
        save_strategy="steps",
        save_steps=1,
        save_total_limit=1,
        logging_strategy="steps",
        logging_steps=1,
        predict_with_generate=False,
        remove_unused_columns=False,
        report_to=[],
        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=0,
        do_train=True,
        do_eval=True,
        optim="adamw_torch",
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=prepared["train"],
        eval_dataset=prepared["validation"],
        data_collator=collator,
        processing_class=processor,
    )

    train_result = trainer.train()
    if not math.isfinite(float(train_result.training_loss)):
        raise RuntimeError(f"Non-finite training loss: {train_result.training_loss}")
    eval_metrics = trainer.evaluate()
    eval_loss = float(eval_metrics.get("eval_loss", float("nan")))
    if not math.isfinite(eval_loss):
        raise RuntimeError(f"Non-finite evaluation loss: {eval_loss}")

    adapter_dir = args.output_dir / "final_adapter"
    model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    generation = generate_validation_example(model, processor, dataset["validation"][0])

    summary = {
        "status": "PASS",
        "model": args.model,
        "train_record_ids": [record["id"] for record in train_records],
        "validation_record_ids": [record["id"] for record in validation_records],
        "max_steps": args.max_steps,
        "precision": args.precision,
        "training_loss": float(train_result.training_loss),
        "evaluation": eval_metrics,
        "generation": generation,
        "adapter_dir": str(adapter_dir),
    }
    summary_path = args.output_dir / "smoke_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Smoke test: PASS\nSummary: {summary_path}\nAdapter: {adapter_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except torch.cuda.OutOfMemoryError as exc:
        print(
            "CUDA out of memory during the bounded smoke test. Close GPU applications "
            "and retry; do not increase batch size.",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc
