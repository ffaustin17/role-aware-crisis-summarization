"""Fine-tune a T5-small baseline on prepared role-aware summary splits."""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)


DEFAULT_DATA_DIR = Path("data/modeling/t5_baseline_v1")
DEFAULT_OUTPUT_DIR = Path("models/t5_small_baseline_v1")
DEFAULT_MODEL_NAME = "t5-small"
DEFAULT_SEED = 42


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for T5-small baseline training."""
    parser = argparse.ArgumentParser(description="Train the T5-small baseline model.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--max-input-length", type=int, default=512)
    parser.add_argument("--max-target-length", type=int, default=128)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    return parser.parse_args()


def load_split_dataset(data_dir: Path) -> Any:
    """Load prepared train and validation JSONL files."""
    data_files = {
        "train": str(data_dir / "train.jsonl"),
        "validation": str(data_dir / "validation.jsonl"),
    }
    return load_dataset("json", data_files=data_files)


def preprocess_dataset(
    dataset: Any,
    tokenizer: Any,
    max_input_length: int,
    max_target_length: int,
) -> Any:
    """Tokenize input_text and target_text for sequence-to-sequence training."""

    def tokenize_batch(batch: dict[str, list[str]]) -> dict[str, Any]:
        model_inputs = tokenizer(
            batch["input_text"],
            max_length=max_input_length,
            truncation=True,
        )
        labels = tokenizer(
            text_target=batch["target_text"],
            max_length=max_target_length,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )


def build_training_args(args: argparse.Namespace) -> Seq2SeqTrainingArguments:
    """
    Build TrainingArguments with light compatibility across Transformers versions.

    Transformers v5 uses eval_strategy while v4 uses evaluation_strategy.
    """
    signature = inspect.signature(Seq2SeqTrainingArguments.__init__)
    parameter_names = set(signature.parameters)

    training_args: dict[str, Any] = {
        "output_dir": str(args.output_dir),
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.train_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.num_train_epochs,
        "weight_decay": args.weight_decay,
        "save_strategy": "epoch",
        "logging_steps": args.logging_steps,
        "save_total_limit": args.save_total_limit,
        "predict_with_generate": False,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "report_to": "none",
        "seed": args.seed,
        "fp16": torch.cuda.is_available(),
    }

    if "eval_strategy" in parameter_names:
        training_args["eval_strategy"] = "epoch"
    else:
        training_args["evaluation_strategy"] = "epoch"

    return Seq2SeqTrainingArguments(**training_args)


def main() -> int:
    """Train and save the baseline model checkpoint."""
    args = parse_args()
    set_seed(args.seed)

    dataset = load_split_dataset(args.data_dir)
    if args.max_train_samples is not None:
        dataset["train"] = dataset["train"].select(
            range(min(args.max_train_samples, len(dataset["train"])))
        )
    if args.max_eval_samples is not None:
        dataset["validation"] = dataset["validation"].select(
            range(min(args.max_eval_samples, len(dataset["validation"])))
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    tokenized_dataset = preprocess_dataset(
        dataset,
        tokenizer,
        args.max_input_length,
        args.max_target_length,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": build_training_args(args),
        "train_dataset": tokenized_dataset["train"],
        "eval_dataset": tokenized_dataset["validation"],
        "data_collator": data_collator,
    }
    trainer_parameters = set(inspect.signature(Seq2SeqTrainer.__init__).parameters)
    if "processing_class" in trainer_parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Seq2SeqTrainer(**trainer_kwargs)

    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print(f"Saved baseline checkpoint to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
