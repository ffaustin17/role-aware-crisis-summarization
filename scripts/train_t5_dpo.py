"""Preference-optimize the supervised T5 baseline with a custom DPO loop."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup, set_seed


DEFAULT_DATA_DIR = Path("data/preferences")
DEFAULT_TRAIN_FILE = "dpo_preferences_t5_v2_vs_gpt4o_reward_v1_train.jsonl"
DEFAULT_VALIDATION_FILE = "dpo_preferences_t5_v2_vs_gpt4o_reward_v1_validation.jsonl"
DEFAULT_MODEL_NAME_OR_PATH = "models/t5_small_baseline_v2"
DEFAULT_OUTPUT_DIR = Path("models/t5_small_dpo_v1")
DEFAULT_SEED = 42


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for DPO training."""
    parser = argparse.ArgumentParser(description="Train a T5 DPO model.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--train-file", default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--validation-file", default=DEFAULT_VALIDATION_FILE)
    parser.add_argument("--model-name-or-path", default=DEFAULT_MODEL_NAME_OR_PATH)
    parser.add_argument("--reference-model-name-or-path", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-input-length", type=int, default=512)
    parser.add_argument("--max-target-length", type=int, default=128)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--save-every-epoch", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read non-empty JSONL records."""
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} in {path}") from exc
    return records


class PreferenceDataset(Dataset):
    """Dataset wrapper for DPO prompt/chosen/rejected records."""

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        return {
            "tweet_id": record.get("tweet_id"),
            "prompt": record["prompt"],
            "chosen": record["chosen"],
            "rejected": record["rejected"],
            "chosen_model": record.get("chosen_model"),
            "reward_margin": record.get("reward_margin"),
        }


class PreferenceCollator:
    """Tokenize DPO prompts and paired summaries."""

    def __init__(
        self,
        tokenizer: Any,
        max_input_length: int,
        max_target_length: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __call__(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        prompts = [record["prompt"] for record in records]
        chosen = [record["chosen"] for record in records]
        rejected = [record["rejected"] for record in records]

        prompt_tokens = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_input_length,
        )
        chosen_tokens = self.tokenizer(
            text_target=chosen,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_target_length,
        )
        rejected_tokens = self.tokenizer(
            text_target=rejected,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_target_length,
        )

        return {
            "tweet_ids": [record["tweet_id"] for record in records],
            "chosen_models": [record["chosen_model"] for record in records],
            "reward_margins": [record["reward_margin"] for record in records],
            "input_ids": prompt_tokens["input_ids"],
            "attention_mask": prompt_tokens["attention_mask"],
            "chosen_labels": chosen_tokens["input_ids"],
            "chosen_attention_mask": chosen_tokens["attention_mask"],
            "rejected_labels": rejected_tokens["input_ids"],
            "rejected_attention_mask": rejected_tokens["attention_mask"],
        }


def labels_with_ignore_index(labels: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    """Replace padding token IDs with -100 for seq2seq loss/log-prob scoring."""
    labels = labels.clone()
    labels[labels == pad_token_id] = -100
    return labels


def sequence_log_probs(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    pad_token_id: int,
) -> torch.Tensor:
    """Return average per-token log probability for each target sequence."""
    model_labels = labels_with_ignore_index(labels, pad_token_id)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=model_labels,
    )
    logits = outputs.logits
    log_probs = F.log_softmax(logits, dim=-1)
    safe_labels = labels.clone()
    safe_labels[safe_labels == pad_token_id] = 0
    token_log_probs = log_probs.gather(
        dim=-1,
        index=safe_labels.unsqueeze(-1),
    ).squeeze(-1)
    target_mask = labels.ne(pad_token_id)
    summed_log_probs = (token_log_probs * target_mask).sum(dim=-1)
    token_counts = target_mask.sum(dim=-1).clamp_min(1)
    return summed_log_probs / token_counts


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move tensor values in a batch to the selected device."""
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute DPO loss and batch metrics."""
    policy_log_ratios = policy_chosen_logps - policy_rejected_logps
    reference_log_ratios = reference_chosen_logps - reference_rejected_logps
    logits = policy_log_ratios - reference_log_ratios
    losses = -F.logsigmoid(beta * logits)
    rewards_chosen = beta * (policy_chosen_logps - reference_chosen_logps)
    rewards_rejected = beta * (policy_rejected_logps - reference_rejected_logps)
    metrics = {
        "loss": float(losses.mean().detach().cpu()),
        "preference_accuracy": float((logits > 0).float().mean().detach().cpu()),
        "policy_margin": float(policy_log_ratios.mean().detach().cpu()),
        "reference_margin": float(reference_log_ratios.mean().detach().cpu()),
        "implicit_reward_margin": float(
            (rewards_chosen - rewards_rejected).mean().detach().cpu()
        ),
    }
    return losses.mean(), metrics


def evaluate(
    model: Any,
    reference_model: Any,
    data_loader: DataLoader,
    device: torch.device,
    pad_token_id: int,
    beta: float,
) -> dict[str, float]:
    """Evaluate DPO loss on a validation split."""
    model.eval()
    reference_model.eval()
    collected: dict[str, list[float]] = defaultdict_list()
    with torch.no_grad():
        for batch in data_loader:
            batch = move_batch_to_device(batch, device)
            policy_chosen = sequence_log_probs(
                model,
                batch["input_ids"],
                batch["attention_mask"],
                batch["chosen_labels"],
                pad_token_id,
            )
            policy_rejected = sequence_log_probs(
                model,
                batch["input_ids"],
                batch["attention_mask"],
                batch["rejected_labels"],
                pad_token_id,
            )
            reference_chosen = sequence_log_probs(
                reference_model,
                batch["input_ids"],
                batch["attention_mask"],
                batch["chosen_labels"],
                pad_token_id,
            )
            reference_rejected = sequence_log_probs(
                reference_model,
                batch["input_ids"],
                batch["attention_mask"],
                batch["rejected_labels"],
                pad_token_id,
            )
            _, metrics = dpo_loss(
                policy_chosen,
                policy_rejected,
                reference_chosen,
                reference_rejected,
                beta,
            )
            for key, value in metrics.items():
                collected[key].append(value)
    model.train()
    return {f"eval_{key}": mean(values) for key, values in collected.items()}


def defaultdict_list() -> dict[str, list[float]]:
    """Return a regular dict with list defaults for metric collection."""
    return defaultdict(list)


def write_training_log(rows: list[dict[str, Any]], path: Path) -> None:
    """Write training/evaluation metrics as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_checkpoint(model: Any, tokenizer: Any, output_dir: Path) -> None:
    """Save a model/tokenizer checkpoint."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


def load_records(path: Path, max_samples: int | None) -> list[dict[str, Any]]:
    """Load JSONL records and optionally keep a prefix sample."""
    records = read_jsonl(path)
    if max_samples is not None:
        records = records[:max_samples]
    return records


def main() -> int:
    """Run custom seq2seq DPO training."""
    args = parse_args()
    set_seed(args.seed)

    train_path = args.data_dir / args.train_file
    validation_path = args.data_dir / args.validation_file
    train_records = load_records(train_path, args.max_train_samples)
    validation_records = load_records(validation_path, args.max_eval_samples)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path).to(device)
    reference_path = args.reference_model_name_or_path or args.model_name_or_path
    reference_model = AutoModelForSeq2SeqLM.from_pretrained(reference_path).to(device)
    reference_model.eval()
    for parameter in reference_model.parameters():
        parameter.requires_grad_(False)

    collator = PreferenceCollator(
        tokenizer=tokenizer,
        max_input_length=args.max_input_length,
        max_target_length=args.max_target_length,
    )
    train_loader = DataLoader(
        PreferenceDataset(train_records),
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    validation_loader = DataLoader(
        PreferenceDataset(validation_records),
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    update_steps_per_epoch = math.ceil(
        len(train_loader) / args.gradient_accumulation_steps
    )
    total_update_steps = max(1, math.ceil(args.num_train_epochs * update_steps_per_epoch))
    warmup_steps = int(total_update_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_update_steps,
    )

    pad_token_id = tokenizer.pad_token_id
    training_log: list[dict[str, Any]] = []
    model.train()
    global_step = 0
    optimizer.zero_grad(set_to_none=True)

    completed_epochs = int(math.ceil(args.num_train_epochs))
    for epoch in range(completed_epochs):
        if epoch >= args.num_train_epochs:
            break
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        running_metrics: dict[str, list[float]] = defaultdict_list()
        for step, batch in enumerate(progress, start=1):
            batch = move_batch_to_device(batch, device)
            policy_chosen = sequence_log_probs(
                model,
                batch["input_ids"],
                batch["attention_mask"],
                batch["chosen_labels"],
                pad_token_id,
            )
            policy_rejected = sequence_log_probs(
                model,
                batch["input_ids"],
                batch["attention_mask"],
                batch["rejected_labels"],
                pad_token_id,
            )
            with torch.no_grad():
                reference_chosen = sequence_log_probs(
                    reference_model,
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["chosen_labels"],
                    pad_token_id,
                )
                reference_rejected = sequence_log_probs(
                    reference_model,
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["rejected_labels"],
                    pad_token_id,
                )

            loss, metrics = dpo_loss(
                policy_chosen,
                policy_rejected,
                reference_chosen,
                reference_rejected,
                args.beta,
            )
            scaled_loss = loss / args.gradient_accumulation_steps
            scaled_loss.backward()

            for key, value in metrics.items():
                running_metrics[key].append(value)

            should_update = (
                step % args.gradient_accumulation_steps == 0
                or step == len(train_loader)
            )
            if should_update:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % args.logging_steps == 0:
                    log_row = {
                        "epoch": epoch + 1,
                        "step": global_step,
                        "learning_rate": scheduler.get_last_lr()[0],
                    }
                    log_row.update(
                        {
                            f"train_{key}": mean(values)
                            for key, values in running_metrics.items()
                        }
                    )
                    training_log.append(log_row)
                    progress.set_postfix(
                        {
                            "loss": f"{log_row.get('train_loss', 0.0):.4f}",
                            "acc": f"{log_row.get('train_preference_accuracy', 0.0):.3f}",
                        }
                    )
                    running_metrics = defaultdict_list()

        eval_metrics = evaluate(
            model,
            reference_model,
            validation_loader,
            device,
            pad_token_id,
            args.beta,
        )
        eval_row = {"epoch": epoch + 1, "step": global_step}
        eval_row.update(eval_metrics)
        training_log.append(eval_row)
        print(json.dumps(eval_row, indent=2))

        if args.save_every_epoch:
            save_checkpoint(
                model,
                tokenizer,
                args.output_dir / f"checkpoint-epoch-{epoch + 1}",
            )

    save_checkpoint(model, tokenizer, args.output_dir)
    write_training_log(training_log, args.output_dir / "dpo_training_log.jsonl")
    print(f"Train records: {len(train_records)}")
    print(f"Validation records: {len(validation_records)}")
    print(f"Total optimizer steps: {global_step}")
    print(f"Saved DPO checkpoint to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
