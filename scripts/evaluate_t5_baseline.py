"""Evaluate a fine-tuned T5 baseline with ROUGE, BLEU, and BERTScore."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

import sacrebleu
import torch
from bert_score import score as bert_score
from rouge_score import rouge_scorer
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


DEFAULT_DATA_DIR = Path("data/modeling/t5_baseline_v1")
DEFAULT_MODEL_DIR = Path("models/t5_small_baseline_v1")
DEFAULT_METRICS_CSV_PATH = Path("reports/tables/t5_small_baseline_metrics.csv")
DEFAULT_METRICS_MD_PATH = Path("reports/tables/t5_small_baseline_metrics.md")
DEFAULT_PREDICTIONS_PATH = Path("data/modeling/t5_baseline_v1/test_predictions.jsonl")
DEFAULT_BERTSCORE_MODEL = "roberta-large"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for baseline evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate the T5-small baseline.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--metrics-csv", type=Path, default=DEFAULT_METRICS_CSV_PATH)
    parser.add_argument("--metrics-md", type=Path, default=DEFAULT_METRICS_MD_PATH)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS_PATH)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-input-length", type=int, default=512)
    parser.add_argument("--max-generation-length", type=int, default=128)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--bertscore-model", default=DEFAULT_BERTSCORE_MODEL)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read non-empty JSONL records from disk."""
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


def batched(records: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    """Yield fixed-size batches from a record list."""
    return [
        records[index : index + batch_size]
        for index in range(0, len(records), batch_size)
    ]


def generate_predictions(
    records: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    device: torch.device,
    batch_size: int,
    max_input_length: int,
    max_generation_length: int,
    num_beams: int,
) -> list[dict[str, Any]]:
    """Generate summaries for test records."""
    predictions: list[dict[str, Any]] = []
    model.eval()

    for batch in tqdm(batched(records, batch_size), desc="Generating"):
        inputs = [record["input_text"] for record in batch]
        encoded = tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length,
        ).to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                **encoded,
                max_length=max_generation_length,
                num_beams=num_beams,
            )

        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        for record, prediction in zip(batch, decoded, strict=True):
            predictions.append(
                {
                    "tweet_id": record["tweet_id"],
                    "source_row_id": record.get("source_row_id"),
                    "role": record.get("role"),
                    "disaster_type": record.get("disaster_type"),
                    "information_type": record.get("information_type"),
                    "input_text": record["input_text"],
                    "target_text": record["target_text"],
                    "prediction_text": prediction.strip(),
                }
            )

    return predictions


def compute_rouge(predictions: list[str], references: list[str]) -> dict[str, float]:
    """Compute average ROUGE F1 scores."""
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True,
    )
    score_lists: dict[str, list[float]] = {"rouge1": [], "rouge2": [], "rougeL": []}
    for prediction, reference in zip(predictions, references, strict=True):
        scores = scorer.score(reference, prediction)
        for metric_name in score_lists:
            score_lists[metric_name].append(scores[metric_name].fmeasure)

    return {metric_name: mean(values) for metric_name, values in score_lists.items()}


def compute_metrics(
    prediction_records: list[dict[str, Any]],
    bertscore_model: str,
) -> dict[str, float]:
    """Compute ROUGE, BLEU, and BERTScore for generated predictions."""
    predictions = [record["prediction_text"] for record in prediction_records]
    references = [record["target_text"] for record in prediction_records]

    rouge_scores = compute_rouge(predictions, references)
    bleu = sacrebleu.corpus_bleu(predictions, [references]).score
    bert_precision, bert_recall, bert_f1 = bert_score(
        predictions,
        references,
        model_type=bertscore_model,
        lang="en",
        verbose=True,
    )

    return {
        "rouge1_f1": rouge_scores["rouge1"],
        "rouge2_f1": rouge_scores["rouge2"],
        "rougeL_f1": rouge_scores["rougeL"],
        "bleu": bleu,
        "bertscore_precision": float(bert_precision.mean()),
        "bertscore_recall": float(bert_recall.mean()),
        "bertscore_f1": float(bert_f1.mean()),
        "num_examples": float(len(prediction_records)),
    }


def write_predictions(records: list[dict[str, Any]], path: Path) -> None:
    """Write prediction records as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_metrics_csv(metrics: dict[str, float], path: Path) -> None:
    """Write metrics as a two-column CSV table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        for metric_name, value in metrics.items():
            writer.writerow([metric_name, value])


def write_metrics_markdown(metrics: dict[str, float], path: Path) -> None:
    """Write metrics as a Markdown table for reports."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        file.write("| metric | value |\n")
        file.write("|---|---:|\n")
        for metric_name, value in metrics.items():
            file.write(f"| {metric_name} | {value:.6f} |\n")


def main() -> int:
    """Run baseline evaluation and write metrics artifacts."""
    args = parse_args()
    records = read_jsonl(args.data_dir / "test.jsonl")
    if args.max_eval_samples is not None:
        records = records[: args.max_eval_samples]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir).to(device)

    prediction_records = generate_predictions(
        records=records,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=args.batch_size,
        max_input_length=args.max_input_length,
        max_generation_length=args.max_generation_length,
        num_beams=args.num_beams,
    )
    metrics = compute_metrics(prediction_records, args.bertscore_model)

    write_predictions(prediction_records, args.predictions)
    write_metrics_csv(metrics, args.metrics_csv)
    write_metrics_markdown(metrics, args.metrics_md)

    print(f"Evaluated {len(prediction_records)} examples")
    print(f"Wrote predictions to: {args.predictions}")
    print(f"Wrote metrics CSV to: {args.metrics_csv}")
    print(f"Wrote metrics Markdown to: {args.metrics_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
