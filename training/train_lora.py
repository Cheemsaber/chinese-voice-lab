"""Train a configuration-driven Whisper LoRA adapter on the full train split."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
import warnings
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, PeftModelForSeq2SeqLM, TaskType
from transformers import (
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

from lora_common import (
    DataCollatorSpeechSeq2SeqWithPadding,
    ValidationMetricRecorder,
    apply_run_overrides,
    collect_run_metadata,
    effective_batch_size,
    load_config,
    load_full_splits,
    load_processor_and_model,
    prepare_run_directory,
    prepare_trainer_dataset,
    run_audio_preflight,
    utc_now,
    validate_runtime,
    write_json,
    write_reproducibility_artifacts,
)


class WhisperPeftModelForSeq2SeqLM(PeftModelForSeq2SeqLM):
    """Keep seq2seq generation while forwarding Whisper audio arguments unchanged.

    PEFT 0.19.1's generic seq2seq forward injects ``input_ids=None`` for a text
    encoder. Transformers 5.14 Whisper instead receives ``input_features`` and
    forwards extra keyword arguments to its decoder, where the injected
    ``input_ids`` collides with ``decoder_input_ids``. LoRA itself needs only
    PEFT's hooks, so the generic PeftModel forward is the correct audio path.
    """

    def forward(
        self,
        input_features: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        decoder_input_ids: torch.Tensor | None = None,
        decoder_attention_mask: torch.Tensor | None = None,
        encoder_outputs: Any = None,
        past_key_values: Any = None,
        decoder_inputs_embeds: torch.Tensor | None = None,
        decoder_position_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Any,
    ) -> Any:
        return PeftModel.forward(
            self,
            input_features=input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            decoder_position_ids=decoder_position_ids,
            labels=labels,
            use_cache=use_cache,
            **kwargs,
        )

    def generate(self, **kwargs: Any) -> Any:
        input_features = kwargs.get("input_features")
        if input_features is not None:
            encoder_dtype = self.get_base_model().model.encoder.conv1.weight.dtype
            kwargs["input_features"] = input_features.to(dtype=encoder_dtype)
        return super().generate(**kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Whisper LoRA adapter from a validated YAML configuration."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Permit Hugging Face downloads. Offline/local-only loading is the default.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        help="Resume only after the requested configuration matches resolved_config.yaml.",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Archive an existing run directory before starting a fresh run.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate configuration, manifest, split isolation, and all audio without loading a model.",
    )
    return parser.parse_args()


def print_run_plan(config: dict[str, Any], splits: dict[str, list[dict[str, Any]]]) -> None:
    training = config["training"]
    print("Resolved training plan:")
    print(f"  Run:                    {config['run']['name']}")
    print(f"  Output:                 {config['run']['output_dir']}")
    print(f"  Model:                  {config['model']['id']}")
    print(f"  Precision:              {config['model']['dtype'].upper()}")
    print(
        "  Records:                "
        f"train={len(splits['train'])}, validation={len(splits['validation'])}"
    )
    print(f"  Train micro-batch:      {training['per_device_train_batch_size']}")
    print(f"  Gradient accumulation:  {training['gradient_accumulation_steps']}")
    print(f"  Effective batch/GPU:    {effective_batch_size(config)}")
    print(f"  Learning rate:          {float(training['learning_rate']):.8g}")
    print(f"  Epochs:                 {training['num_train_epochs']}")
    print(
        "  LoRA:                   "
        f"r={config['lora']['r']}, alpha={config['lora']['lora_alpha']}, "
        f"alpha/r={config['lora']['lora_alpha'] / config['lora']['r']:.1f}, "
        f"dropout={config['lora']['lora_dropout']}, "
        f"targets={config['lora']['target_modules']}"
    )


def build_lora_model(config: dict[str, Any], base_model: torch.nn.Module) -> torch.nn.Module:
    values = config["lora"]
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=int(values["r"]),
        lora_alpha=int(values["lora_alpha"]),
        lora_dropout=float(values["lora_dropout"]),
        target_modules=list(values["target_modules"]),
        bias=str(values["bias"]),
    )
    model = WhisperPeftModelForSeq2SeqLM(base_model, lora_config)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.print_trainable_parameters()
    return model


def build_training_arguments(
    config: dict[str, Any], output_dir: Path
) -> Seq2SeqTrainingArguments:
    training_values = dict(config["training"])
    training_values["warmup_steps"] = training_values.pop("warmup_ratio")
    report_to = training_values.get("report_to")
    if report_to and report_to != "none" and report_to != []:
        os.environ["TENSORBOARD_LOGGING_DIR"] = str(output_dir / "tensorboard")
    return Seq2SeqTrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        run_name=str(config["run"]["name"]),
        do_train=True,
        do_eval=True,
        **training_values,
    )


def train(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    config = apply_run_overrides(
        config,
        resume_from_checkpoint=args.resume_from_checkpoint,
        overwrite_output=args.overwrite_output,
    )
    set_seed(int(config["training"]["seed"]))
    _, splits = load_full_splits(config)
    print_run_plan(config, splits)
    run_audio_preflight(splits, int(config["data"]["sampling_rate"]))
    if args.validate_only:
        print("Full-data configuration and audio preflight: PASS")
        return 0

    validate_runtime(config, require_cuda=True)
    output_dir = prepare_run_directory(config)
    repo_root = Path(__file__).resolve().parents[1]
    metadata = collect_run_metadata(config, splits, repo_root)
    write_reproducibility_artifacts(output_dir, config, metadata)

    try:
        processor, base_model = load_processor_and_model(config, args.allow_download)
        metadata["model"]["cached_revision"] = getattr(
            base_model.config, "_commit_hash", None
        )
        write_json(output_dir / "run_metadata.json", metadata)

        model = build_lora_model(config, base_model)
        warnings.filterwarnings(
            "ignore",
            message=(
                r"Could not find a config file in .* - will assume that the "
                r"vocabulary was not modified\."
            ),
            module=r"peft\.utils\.save_and_load",
        )
        prepared = prepare_trainer_dataset(
            splits,
            processor,
            int(config["data"]["sampling_rate"]),
        )
        collator = DataCollatorSpeechSeq2SeqWithPadding(
            processor=processor,
            decoder_start_token_id=model.config.decoder_start_token_id,
        )
        metric_recorder = ValidationMetricRecorder(
            processor=processor,
            records=splits["validation"],
            evaluation=config["evaluation"],
            output_dir=output_dir,
        )
        callbacks = []
        if config["early_stopping"]["enabled"]:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=int(
                        config["early_stopping"]["patience"]
                    ),
                    early_stopping_threshold=float(
                        config["early_stopping"]["threshold"]
                    ),
                )
            )
        training_args = build_training_arguments(config, output_dir)
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=prepared["train"],
            eval_dataset=prepared["validation"],
            data_collator=collator,
            processing_class=processor,
            compute_metrics=metric_recorder,
            callbacks=callbacks,
        )

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        resume_value = config["run"]["resume_from_checkpoint"]
        train_result = trainer.train(
            resume_from_checkpoint=str(resume_value) if resume_value else None
        )
        train_loss = float(train_result.metrics.get("train_loss", float("nan")))
        if not math.isfinite(train_loss):
            raise RuntimeError(f"Non-finite training loss: {train_loss}")
        trainer.save_metrics("train", train_result.metrics)
        write_json(output_dir / "train_metrics.json", train_result.metrics)

        final_evaluation = trainer.evaluate(metric_key_prefix="eval")
        eval_loss = float(final_evaluation.get("eval_loss", float("nan")))
        if not math.isfinite(eval_loss):
            raise RuntimeError(f"Non-finite evaluation loss: {eval_loss}")
        trainer.save_metrics("eval", final_evaluation)
        trainer.state.save_to_json(str(output_dir / "trainer_state.json"))

        adapter_dir = output_dir / "best_adapter"
        trainer.model.save_pretrained(adapter_dir)
        processor.save_pretrained(adapter_dir)

        metadata["status"] = "PASS"
        metadata["finished_at"] = utc_now()
        metadata["best_checkpoint"] = trainer.state.best_model_checkpoint
        metadata["best_metric"] = trainer.state.best_metric
        metadata["adapter_dir"] = str(adapter_dir)
        if torch.cuda.is_available():
            metadata["peak_cuda_memory"] = {
                "allocated_bytes": torch.cuda.max_memory_allocated(),
                "reserved_bytes": torch.cuda.max_memory_reserved(),
            }
        write_json(output_dir / "run_metadata.json", metadata)
        write_json(
            output_dir / "run_summary.json",
            {
                "status": "PASS",
                "run_name": config["run"]["name"],
                "model": config["model"]["id"],
                "learning_rate": config["training"]["learning_rate"],
                "lora": config["lora"],
                "train_metrics": train_result.metrics,
                "final_evaluation": final_evaluation,
                "best_checkpoint": trainer.state.best_model_checkpoint,
                "best_metric": trainer.state.best_metric,
                "adapter_dir": str(adapter_dir),
            },
        )
        print(f"Training: PASS\nRun directory: {output_dir}\nAdapter: {adapter_dir}")
        return 0
    except Exception as exc:
        metadata["status"] = "FAILED"
        metadata["finished_at"] = utc_now()
        metadata["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        if torch.cuda.is_available():
            metadata["peak_cuda_memory"] = {
                "allocated_bytes": torch.cuda.max_memory_allocated(),
                "reserved_bytes": torch.cuda.max_memory_reserved(),
            }
        write_json(output_dir / "run_metadata.json", metadata)
        raise


def main() -> int:
    return train(parse_args())


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except torch.cuda.OutOfMemoryError as exc:
        print(
            "CUDA out of memory. Reduce per_device_train_batch_size, increase "
            "gradient_accumulation_steps to preserve the effective batch, and retry "
            "with a new run directory.",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc
