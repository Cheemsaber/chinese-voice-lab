"""Evaluate a frozen Whisper base model or LoRA adapter on one complete split."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch
from peft import PeftModel

from lora_common import (
    build_audio_dataset,
    compute_transcription_metrics,
    decode_audio,
    load_config,
    load_full_splits,
    load_processor_and_model,
    torch_dtype,
    utc_now,
    validate_runtime,
    write_json,
    write_jsonl,
    normalize_transcript,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a Whisper base model or LoRA adapter on a complete split."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--adapter",
        type=Path,
        help="Adapter directory. Omit this argument for the frozen base-model baseline.",
    )
    parser.add_argument(
        "--split", choices=("validation", "test"), default="validation"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional result directory override. Defaults inside this config's run directory.",
    )
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-") or "model"


def resolve_output_dir(
    config: dict, adapter: Path | None, requested: Path | None
) -> Path:
    if requested is not None:
        return requested.resolve()
    run_output = Path(str(config["run"]["output_dir"])).resolve()
    if adapter is None:
        return run_output.parent / "baselines" / safe_name(str(config["run"]["name"]))
    return run_output / "evaluation" / safe_name(adapter.resolve().name)


def evaluate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    validate_runtime(config, require_cuda=True)
    _, splits = load_full_splits(config, include_test=args.split == "test")
    records = splits[args.split]
    output_dir = resolve_output_dir(config, args.adapter, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processor, base_model = load_processor_and_model(config, args.allow_download)
    model: torch.nn.Module = base_model
    adapter_path: Path | None = None
    if args.adapter is not None:
        adapter_path = args.adapter.resolve()
        if not adapter_path.is_dir():
            raise FileNotFoundError(f"Adapter directory does not exist: {adapter_path}")
        model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=False)

    device = torch.device("cuda")
    dtype = torch_dtype(str(config["model"]["dtype"]))
    model.to(device)
    model.eval()
    model.config.use_cache = True
    dataset = build_audio_dataset(records, int(config["data"]["sampling_rate"]))

    predictions: list[str] = []
    references: list[str] = []
    record_ids: list[str] = []
    generation_max_length = int(config["training"]["generation_max_length"])
    with torch.inference_mode():
        for index, record in enumerate(dataset, 1):
            waveform, sample_rate = decode_audio(record["audio"])
            inputs = processor.feature_extractor(
                waveform.numpy(), sampling_rate=sample_rate, return_tensors="pt"
            )
            generated = model.generate(
                input_features=inputs.input_features.to(device=device, dtype=dtype),
                max_length=generation_max_length,
                language=config["model"]["language_token"],
                task=config["model"]["task"],
            )
            raw_prediction = processor.batch_decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            prediction = normalize_transcript(raw_prediction)
            predictions.append(prediction)
            references.append(record["text"])
            record_ids.append(record["id"])
            print(f"[{index}/{len(dataset)}] {record['id']}: {prediction}")

    metrics, rows = compute_transcription_metrics(
        record_ids,
        references,
        predictions,
        config["evaluation"],
    )
    if config["evaluation"]["save_predictions"]:
        write_jsonl(output_dir / f"{args.split}_predictions.jsonl", rows)
    write_json(output_dir / f"{args.split}_metrics.json", metrics)
    write_json(
        output_dir / f"{args.split}_evaluation_metadata.json",
        {
            "evaluated_at": utc_now(),
            "config": str(args.config.resolve()),
            "run_name": config["run"]["name"],
            "model": config["model"]["id"],
            "adapter": str(adapter_path) if adapter_path else None,
            "split": args.split,
            "record_count": len(records),
            "metrics": metrics,
        },
    )
    print(f"Evaluation: PASS\nResults: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(evaluate(parse_args()))
