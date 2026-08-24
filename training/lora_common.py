"""Shared configuration, dataset, model, metric, and artifact helpers for LoRA runs."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Iterable, Sequence

import unicodedata
import opencc

import jiwer
import torch
import yaml
from datasets import Audio, Dataset, DatasetDict
from transformers import WhisperForConditionalGeneration, WhisperProcessor

try:
    from .smoke_test import (
        DataCollatorSpeechSeq2SeqWithPadding,
        assert_split_isolation,
        decode_audio,
        load_and_validate_manifest,
    )
except ImportError:
    from smoke_test import (  # type: ignore[no-redef]
        DataCollatorSpeechSeq2SeqWithPadding,
        assert_split_isolation,
        decode_audio,
        load_and_validate_manifest,
    )


TOP_LEVEL_KEYS = {
    "run",
    "data",
    "model",
    "lora",
    "training",
    "early_stopping",
    "evaluation",
}
SECTION_KEYS = {
    "run": {"name", "output_dir", "overwrite_output", "resume_from_checkpoint"},
    "data": {
        "dataset_root",
        "manifest",
        "train_split",
        "validation_split",
        "test_split",
        "audio_field",
        "text_field",
        "sampling_rate",
    },
    "model": {
        "id",
        "language",
        "language_token",
        "task",
        "local_files_only",
        "dtype",
        "use_safetensors",
        "low_cpu_mem_usage",
        "attn_implementation",
    },
    "lora": {
        "task_type",
        "r",
        "lora_alpha",
        "lora_dropout",
        "target_modules",
        "bias",
    },
    "early_stopping": {"enabled", "patience", "threshold"},
    "evaluation": {
        "identifier_pattern",
        "normalize_uppercase",
        "deduplicate_identifiers",
        "save_predictions",
    },
}
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
REQUIRED_SECTION_KEYS = {
    "run": {"name", "output_dir", "overwrite_output", "resume_from_checkpoint"},
    "data": {
        "dataset_root",
        "manifest",
        "train_split",
        "validation_split",
        "test_split",
        "audio_field",
        "text_field",
        "sampling_rate",
    },
    "model": {
        "id",
        "language",
        "language_token",
        "task",
        "local_files_only",
        "dtype",
        "use_safetensors",
        "low_cpu_mem_usage",
        "attn_implementation",
    },
    "lora": {
        "task_type",
        "r",
        "lora_alpha",
        "lora_dropout",
        "target_modules",
        "bias",
    },
    "early_stopping": {"enabled", "patience", "threshold"},
    "evaluation": {
        "identifier_pattern",
        "normalize_uppercase",
        "deduplicate_identifiers",
        "save_predictions",
    },
}

TRADITIONAL_TO_SIMPLIFIED = opencc.OpenCC("t2s.json")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a YAML mapping")
    return value


def _require_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be an integer of at least 1")
    return value


def _require_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def load_config(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = _require_mapping(loaded, "configuration")
    validate_config(config)
    return copy.deepcopy(config)


def validate_config(config: dict[str, Any]) -> None:
    missing_sections = sorted(TOP_LEVEL_KEYS - set(config))
    unknown_sections = sorted(set(config) - TOP_LEVEL_KEYS)
    if missing_sections:
        raise ValueError(f"Configuration is missing sections: {missing_sections}")
    if unknown_sections:
        raise ValueError(f"Configuration has unknown sections: {unknown_sections}")

    for section_name in TOP_LEVEL_KEYS:
        section = _require_mapping(config[section_name], section_name)
        allowed = (
            ALLOWED_TRAINING_KEYS
            if section_name == "training"
            else SECTION_KEYS[section_name]
        )
        required = (
            ALLOWED_TRAINING_KEYS
            if section_name == "training"
            else REQUIRED_SECTION_KEYS[section_name]
        )
        missing = sorted(required - set(section))
        unknown = sorted(set(section) - allowed)
        if missing:
            raise ValueError(f"{section_name} is missing keys: {missing}")
        if unknown:
            raise ValueError(f"{section_name} has unknown keys: {unknown}")

    run = config["run"]
    if not isinstance(run["name"], str) or not run["name"].strip():
        raise ValueError("run.name must be a nonempty string")
    output_dir = Path(str(run["output_dir"]))
    if output_dir.name.casefold() != run["name"].casefold():
        raise ValueError("run.output_dir must end with run.name to isolate run artifacts")
    if not isinstance(run["overwrite_output"], bool):
        raise ValueError("run.overwrite_output must be true or false")
    if run["resume_from_checkpoint"] is not None and not isinstance(
        run["resume_from_checkpoint"], str
    ):
        raise ValueError("run.resume_from_checkpoint must be null or a path string")

    data = config["data"]
    split_names = [data["train_split"], data["validation_split"], data["test_split"]]
    if any(not isinstance(value, str) or not value for value in split_names):
        raise ValueError("data split names must be nonempty strings")
    if len(set(split_names)) != 3:
        raise ValueError("train, validation, and test split names must be distinct")
    if data["audio_field"] != "audio" or data["text_field"] != "text":
        raise ValueError("The current validated manifest requires audio_field=audio and text_field=text")
    _require_positive_int(data["sampling_rate"], "data.sampling_rate")

    model = config["model"]
    for key in ("id", "language", "language_token", "task", "attn_implementation"):
        if not isinstance(model[key], str) or not model[key].strip():
            raise ValueError(f"model.{key} must be a nonempty string")
    if model["dtype"] not in {"fp32", "fp16", "bf16"}:
        raise ValueError("model.dtype must be one of fp32, fp16, or bf16")
    for key in ("local_files_only", "use_safetensors", "low_cpu_mem_usage"):
        if not isinstance(model[key], bool):
            raise ValueError(f"model.{key} must be true or false")

    lora = config["lora"]
    if lora["task_type"] != "SEQ_2_SEQ_LM":
        raise ValueError("lora.task_type must be SEQ_2_SEQ_LM")
    rank = _require_positive_int(lora["r"], "lora.r")
    alpha = _require_positive_int(lora["lora_alpha"], "lora.lora_alpha")
    if alpha != 2 * rank:
        raise ValueError(
            f"LoRA alpha/r must remain 2; received alpha={alpha}, r={rank}"
        )
    dropout = _require_number(lora["lora_dropout"], "lora.lora_dropout")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("lora.lora_dropout must be in [0, 1)")
    targets = lora["target_modules"]
    if not isinstance(targets, list) or not targets or not all(
        isinstance(value, str) and value for value in targets
    ):
        raise ValueError("lora.target_modules must be a nonempty string list")
    if len(set(targets)) != len(targets):
        raise ValueError("lora.target_modules must not contain duplicates")
    if lora["bias"] not in {"none", "all", "lora_only"}:
        raise ValueError("lora.bias must be none, all, or lora_only")

    training = config["training"]
    for key in (
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
        "gradient_accumulation_steps",
        "logging_steps",
        "save_total_limit",
        "dataloader_num_workers",
        "seed",
        "data_seed",
    ):
        value = training[key]
        if key == "dataloader_num_workers":
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("training.dataloader_num_workers must be nonnegative")
        else:
            _require_positive_int(value, f"training.{key}")
    learning_rate = _require_number(training["learning_rate"], "training.learning_rate")
    if learning_rate <= 0:
        raise ValueError("training.learning_rate must be positive")
    if learning_rate > 1.0:
        raise ValueError(
            "training.learning_rate must not exceed 1.0; use 0.0001 for 1e-4"
        )
    warmup_ratio = _require_number(training["warmup_ratio"], "training.warmup_ratio")
    if not 0.0 <= warmup_ratio <= 1.0:
        raise ValueError("training.warmup_ratio must be between 0 and 1")
    if _require_number(training["num_train_epochs"], "training.num_train_epochs") <= 0:
        raise ValueError("training.num_train_epochs must be positive")
    if bool(training["bf16"]) == bool(training["fp16"]) and training["bf16"]:
        raise ValueError("Only one of training.bf16 and training.fp16 may be true")
    for key in (
        "bf16",
        "fp16",
        "gradient_checkpointing",
        "predict_with_generate",
        "load_best_model_at_end",
        "greater_is_better",
        "remove_unused_columns",
    ):
        if not isinstance(training[key], bool):
            raise ValueError(f"training.{key} must be true or false")
    if not isinstance(training["gradient_checkpointing_kwargs"], dict):
        raise ValueError("training.gradient_checkpointing_kwargs must be a mapping")
    _require_positive_int(
        training["generation_max_length"], "training.generation_max_length"
    )
    for key in (
        "lr_scheduler_type",
        "eval_strategy",
        "save_strategy",
        "logging_strategy",
        "metric_for_best_model",
        "optim",
    ):
        if not isinstance(training[key], str) or not training[key]:
            raise ValueError(f"training.{key} must be a nonempty string")
    report_to = training["report_to"]
    if not isinstance(report_to, list) or not all(
        isinstance(value, str) and value for value in report_to
    ):
        raise ValueError("training.report_to must be a list of strings")
    expected_dtype = "bf16" if training["bf16"] else "fp16" if training["fp16"] else "fp32"
    if model["dtype"] != expected_dtype:
        raise ValueError(
            f"model.dtype={model['dtype']} does not match training precision {expected_dtype}"
        )
    if training["load_best_model_at_end"] and (
        training["eval_strategy"] != training["save_strategy"]
    ):
        raise ValueError(
            "training.eval_strategy and save_strategy must match when loading the best model"
        )
    if training["metric_for_best_model"] != "identifier_exact_accuracy":
        raise ValueError(
            "training.metric_for_best_model must be identifier_exact_accuracy"
        )
    if not training["predict_with_generate"]:
        raise ValueError("training.predict_with_generate must be true for transcription metrics")

    early = config["early_stopping"]
    if not isinstance(early["enabled"], bool):
        raise ValueError("early_stopping.enabled must be true or false")
    _require_positive_int(early["patience"], "early_stopping.patience")
    if _require_number(early["threshold"], "early_stopping.threshold") < 0:
        raise ValueError("early_stopping.threshold must be nonnegative")
    if early["enabled"] and not training["load_best_model_at_end"]:
        raise ValueError("Early stopping requires training.load_best_model_at_end=true")

    evaluation = config["evaluation"]
    try:
        re.compile(str(evaluation["identifier_pattern"]))
    except re.error as exc:
        raise ValueError(f"Invalid evaluation.identifier_pattern: {exc}") from exc
    for key in ("normalize_uppercase", "deduplicate_identifiers", "save_predictions"):
        if not isinstance(evaluation[key], bool):
            raise ValueError(f"evaluation.{key} must be true or false")


def apply_run_overrides(
    config: dict[str, Any],
    resume_from_checkpoint: Path | None,
    overwrite_output: bool,
) -> dict[str, Any]:
    resolved = copy.deepcopy(config)
    if resume_from_checkpoint is not None:
        resolved["run"]["resume_from_checkpoint"] = str(resume_from_checkpoint.resolve())
    if overwrite_output:
        resolved["run"]["overwrite_output"] = True
    validate_config(resolved)
    return resolved


def comparable_config(config: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(config)
    result["run"]["resume_from_checkpoint"] = None
    result["run"]["overwrite_output"] = False
    return result


def prepare_run_directory(config: dict[str, Any]) -> Path:
    output_dir = Path(str(config["run"]["output_dir"])).resolve()
    resume_value = config["run"]["resume_from_checkpoint"]
    overwrite = bool(config["run"]["overwrite_output"])
    saved_config_path = output_dir / "resolved_config.yaml"

    if resume_value:
        checkpoint = Path(str(resume_value)).resolve()
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {checkpoint}")
        if output_dir not in checkpoint.parents:
            raise ValueError("Resume checkpoint must be inside run.output_dir")
        if not saved_config_path.is_file():
            raise FileNotFoundError(
                f"Cannot verify resume configuration; missing {saved_config_path}"
            )
        saved = yaml.safe_load(saved_config_path.read_text(encoding="utf-8"))
        if comparable_config(saved) != comparable_config(config):
            raise ValueError("Requested configuration differs from the saved run configuration")
        return output_dir

    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. "
                "Use a new run.name/output_dir or explicitly enable overwrite_output."
            )
        archive_suffix = datetime.now().strftime("%Y%m%d-%H%M%S")
        archive = output_dir.with_name(f"{output_dir.name}.archive-{archive_suffix}")
        counter = 1
        while archive.exists():
            archive = output_dir.with_name(
                f"{output_dir.name}.archive-{archive_suffix}-{counter}"
            )
            counter += 1
        output_dir.rename(archive)
        print(f"Archived previous output directory to: {archive}")

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_config_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return output_dir


def load_full_splits(
    config: dict[str, Any],
    *,
    include_test: bool = False,
    require_test: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    if require_test and not include_test:
        raise ValueError("require_test=true requires include_test=true")
    data = config["data"]
    dataset_root = Path(str(data["dataset_root"]))
    manifest = Path(str(data["manifest"]))
    records = load_and_validate_manifest(
        manifest,
        dataset_root,
        allow_empty_device_ids=True,
    )
    assert_split_isolation(records)

    split_mapping = {
        "train": str(data["train_split"]),
        "validation": str(data["validation_split"]),
    }
    if include_test:
        split_mapping["test"] = str(data["test_split"])
    splits = {
        purpose: sorted(
            (record for record in records if record["split"] == manifest_name),
            key=lambda record: record["id"],
        )
        for purpose, manifest_name in split_mapping.items()
    }
    required_splits = {"train", "validation"}
    if require_test:
        required_splits.add("test")
    empty = [
        name
        for name, values in splits.items()
        if name in required_splits and not values
    ]
    if empty:
        raise ValueError(f"Required dataset splits are empty: {empty}")
    split_ids = [{record["id"] for record in values} for values in splits.values()]
    if any(
        split_ids[i] & split_ids[j]
        for i in range(len(split_ids))
        for j in range(i + 1, len(split_ids))
    ):
        raise AssertionError("Record IDs overlap across dataset splits")
    return records, splits


def build_audio_dataset(records: Sequence[dict[str, Any]], sampling_rate: int) -> Dataset:
    return Dataset.from_list(list(records)).cast_column(
        "audio", Audio(sampling_rate=sampling_rate)
    )


def run_audio_preflight(
    splits: dict[str, list[dict[str, Any]]], sampling_rate: int
) -> None:
    print("Dataset preflight:")
    split_roles = {
        "train": "optimization",
        "validation": "model selection",
        "test": "held-out final evaluation",
    }
    for split_name in ("train", "validation", "test"):
        if not splits.get(split_name):
            continue
        dataset = build_audio_dataset(splits[split_name], sampling_rate)
        total_seconds = 0.0
        for record in dataset:
            waveform, decoded_rate = decode_audio(record["audio"])
            if decoded_rate != sampling_rate:
                raise ValueError(
                    f"{record['id']}: expected {sampling_rate} Hz, got {decoded_rate}"
                )
            total_seconds += waveform.numel() / decoded_rate
        print(
            f"  {split_name}: {len(dataset)} records, "
            f"{total_seconds / 60:.3f} minutes ({split_roles[split_name]})"
        )


def prepare_trainer_dataset(
    splits: dict[str, list[dict[str, Any]]],
    processor: WhisperProcessor,
    sampling_rate: int,
) -> DatasetDict:
    split_names = [
        name for name in ("train", "validation", "test") if splits.get(name)
    ]
    raw = DatasetDict(
        {
            name: build_audio_dataset(splits[name], sampling_rate)
            for name in split_names
        }
    )

    def prepare_record(record: dict[str, Any]) -> dict[str, Any]:
        waveform, decoded_rate = decode_audio(record["audio"])
        features = processor.feature_extractor(
            waveform.numpy(), sampling_rate=decoded_rate
        ).input_features[0]
        labels = processor.tokenizer(record["text"]).input_ids
        return {"input_features": features, "labels": labels}

    prepared = DatasetDict()
    for split_name in split_names:
        prepared[split_name] = raw[split_name].map(
            prepare_record,
            remove_columns=raw[split_name].column_names,
            keep_in_memory=True,
            load_from_cache_file=False,
            desc=f"Preparing full {split_name} split",
        )
    return prepared


def torch_dtype(dtype_name: str) -> torch.dtype:
    return {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[dtype_name]


def validate_runtime(config: dict[str, Any], require_cuda: bool = True) -> None:
    dtype_name = str(config["model"]["dtype"])
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; full LoRA training requires a CUDA GPU")
    if dtype_name in {"fp16", "bf16"} and not torch.cuda.is_available():
        raise RuntimeError(f"{dtype_name.upper()} requires CUDA")
    if dtype_name == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 was requested, but this GPU does not support BF16")


def load_processor_and_model(
    config: dict[str, Any], allow_download: bool
) -> tuple[WhisperProcessor, WhisperForConditionalGeneration]:
    model_values = config["model"]
    local_files_only = bool(model_values["local_files_only"]) and not allow_download
    print(
        f"Loading {model_values['id']} "
        f"(local_files_only={local_files_only}) from "
        f"HF_HOME={os.environ.get('HF_HOME', Path.home() / '.cache' / 'huggingface')}"
    )
    processor = WhisperProcessor.from_pretrained(
        model_values["id"],
        language=model_values["language"],
        task=model_values["task"],
        local_files_only=local_files_only,
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        model_values["id"],
        dtype=torch_dtype(model_values["dtype"]),
        low_cpu_mem_usage=bool(model_values["low_cpu_mem_usage"]),
        use_safetensors=bool(model_values["use_safetensors"]),
        attn_implementation=model_values["attn_implementation"],
        local_files_only=local_files_only,
    )
    model.config.use_cache = False
    model.generation_config.language = model_values["language_token"]
    model.generation_config.task = model_values["task"]
    return processor, model


def _deduplicate(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def extract_identifiers(text: str, evaluation: dict[str, Any]) -> list[str]:
    normalized = text.upper() if evaluation["normalize_uppercase"] else text
    matches = re.findall(str(evaluation["identifier_pattern"]), normalized)
    values = [match if isinstance(match, str) else "".join(match) for match in matches]
    return _deduplicate(values) if evaluation["deduplicate_identifiers"] else values


def levenshtein_distance(reference: str, hypothesis: str) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, reference_char in enumerate(reference, 1):
        current = [row]
        for column, hypothesis_char in enumerate(hypothesis, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (reference_char != hypothesis_char),
                )
            )
        previous = current
    return previous[-1]


def compute_transcription_metrics(
    record_ids: Sequence[str],
    references: Sequence[str],
    predictions: Sequence[str],
    evaluation: dict[str, Any],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    if not (len(record_ids) == len(references) == len(predictions)):
        raise ValueError("Record IDs, references, and predictions must have equal lengths")
    rows: list[dict[str, Any]] = []
    exact_count = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0
    character_scores: list[float] = []

    for record_id, reference, prediction in zip(record_ids, references, predictions):
        expected_ids = extract_identifiers(reference, evaluation)
        predicted_ids = extract_identifiers(prediction, evaluation)
        exact = predicted_ids == expected_ids
        exact_count += int(exact)
        expected_set = set(expected_ids)
        predicted_set = set(predicted_ids)
        true_positive += len(expected_set & predicted_set)
        false_positive += len(predicted_set - expected_set)
        false_negative += len(expected_set - predicted_set)
        expected_chars = " ".join(expected_ids)
        predicted_chars = " ".join(predicted_ids)
        denominator = max(len(expected_chars), len(predicted_chars), 1)
        character_scores.append(
            max(
                0.0,
                1.0
                - levenshtein_distance(expected_chars, predicted_chars) / denominator,
            )
        )
        rows.append(
            {
                "id": record_id,
                "reference": reference,
                "prediction": prediction,
                "expected_device_ids": expected_ids,
                "predicted_device_ids": predicted_ids,
                "identifier_exact": exact,
            }
        )

    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    metrics = {
        "identifier_exact_accuracy": exact_count / max(len(rows), 1),
        "identifier_character_accuracy": sum(character_scores) / max(len(rows), 1),
        "identifier_precision": (
            true_positive / precision_denominator if precision_denominator else 0.0
        ),
        "identifier_recall": true_positive / recall_denominator if recall_denominator else 0.0,
        "wer": float(jiwer.wer(list(references), list(predictions))),
        "cer": float(jiwer.cer(list(references), list(predictions))),
    }
    return metrics, rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in rows
    )
    path.write_text(content, encoding="utf-8")


@dataclass
class ValidationMetricRecorder:
    processor: WhisperProcessor
    records: Sequence[dict[str, Any]]
    evaluation: dict[str, Any]
    output_dir: Path
    evaluation_count: int = 0

    def __call__(self, eval_prediction: Any) -> dict[str, float]:
        predictions = eval_prediction.predictions
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        if getattr(predictions, "ndim", 0) == 3:
            predictions = predictions.argmax(axis=-1)
        labels = eval_prediction.label_ids.copy()
        labels[labels == -100] = self.processor.tokenizer.pad_token_id
        decoded_predictions = self.processor.batch_decode(
            predictions,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        decoded_predictions = [
            normalize_transcript(text)
            for text in decoded_predictions
        ]
        decoded_references = self.processor.batch_decode(
            labels,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        record_ids = [record["id"] for record in self.records]
        metrics, rows = compute_transcription_metrics(
            record_ids,
            decoded_references,
            decoded_predictions,
            self.evaluation,
        )
        self.evaluation_count += 1
        history_dir = self.output_dir / "validation_history"
        stem = f"evaluation_{self.evaluation_count:04d}"
        if self.evaluation["save_predictions"]:
            write_jsonl(history_dir / f"{stem}_predictions.jsonl", rows)
            write_jsonl(self.output_dir / "validation_predictions.jsonl", rows)
        write_json(history_dir / f"{stem}_metrics.json", metrics)
        write_json(self.output_dir / "validation_metrics.json", metrics)
        return metrics


def manifest_sha256(manifest: Path) -> str:
    digest = hashlib.sha256()
    with manifest.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(repo_root: Path, *arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def collect_run_metadata(
    config: dict[str, Any],
    splits: dict[str, list[dict[str, Any]]],
    repo_root: Path,
) -> dict[str, Any]:
    package_names = [
        "torch",
        "transformers",
        "datasets",
        "accelerate",
        "peft",
        "PyYAML",
        "jiwer",
        "torchcodec",
    ]
    packages: dict[str, str | None] = {}
    for package_name in package_names:
        try:
            packages[package_name] = importlib_metadata.version(package_name)
        except importlib_metadata.PackageNotFoundError:
            packages[package_name] = None
    status = git_value(repo_root, "status", "--porcelain")
    gpu: dict[str, Any] | None = None
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "vram_bytes": properties.total_memory,
            "compute_capability": list(torch.cuda.get_device_capability(0)),
        }
    return {
        "status": "RUNNING",
        "started_at": utc_now(),
        "finished_at": None,
        "git": {
            "commit": git_value(repo_root, "rev-parse", "HEAD"),
            "branch": git_value(repo_root, "branch", "--show-current"),
            "dirty": bool(status),
            "status_porcelain": status.splitlines() if status else [],
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": packages,
            "torch_cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "bf16_supported": (
                torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
            ),
            "gpu": gpu,
        },
        "model": {
            "id": config["model"]["id"],
            "dtype": config["model"]["dtype"],
            "local_files_only": config["model"]["local_files_only"],
            "cached_revision": None,
        },
        "lora": copy.deepcopy(config["lora"]),
        "training": copy.deepcopy(config["training"]),
        "seeds": {
            "seed": config["training"]["seed"],
            "data_seed": config["training"]["data_seed"],
        },
        "records": {
            name: [record["id"] for record in split]
            for name, split in splits.items()
        },
        "peak_cuda_memory": None,
    }


def write_reproducibility_artifacts(
    output_dir: Path,
    config: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    manifest = Path(str(config["data"]["manifest"])).resolve()
    (output_dir / "manifest.sha256").write_text(
        f"{manifest_sha256(manifest)}  {manifest}\n", encoding="utf-8"
    )
    try:
        freeze = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        freeze = f"Unable to capture pip freeze: {exc}\n"
    (output_dir / "requirements_snapshot.txt").write_text(freeze, encoding="utf-8")
    write_json(output_dir / "run_metadata.json", metadata)


def effective_batch_size(config: dict[str, Any], gpu_count: int = 1) -> int:
    training = config["training"]
    return (
        int(training["per_device_train_batch_size"])
        * int(training["gradient_accumulation_steps"])
        * gpu_count
    )

def normalize_transcript(text: str) -> str:
    compatibility_normalized = unicodedata.normalize("NFKC", text)
    return TRADITIONAL_TO_SIMPLIFIED.convert(compatibility_normalized)
