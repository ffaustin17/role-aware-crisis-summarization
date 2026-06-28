"""Analyze role-aware reward score outputs for inspection and calibration."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any


DEFAULT_INPUT_PATH = Path("data/rewards/t5_baseline_v1_reward_scores.jsonl")
DEFAULT_OUTPUT_DIR = Path("reports/tables/reward_analysis")
DEFAULT_EXAMPLE_COUNT = 10

SCORE_COLUMNS = [
    "reward",
    "relevance",
    "factuality",
    "role_coverage",
    "urgency",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for reward-score analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze role-aware reward score JSONL outputs."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--example-count", type=int, default=DEFAULT_EXAMPLE_COUNT)
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


def flatten_category_terms(category_terms: dict[str, list[str]]) -> list[str]:
    """Flatten category-keyed term matches while preserving category context."""
    flattened = []
    for category, terms in category_terms.items():
        for term in terms:
            flattened.append(f"{category}:{term}")
    return flattened


def urgency_terms(urgency_details: dict[str, Any], legacy_key: str, category_key: str) -> list[str]:
    """Return urgency evidence from either legacy term lists or category details."""
    legacy_terms = urgency_details.get(legacy_key)
    if isinstance(legacy_terms, list):
        return [str(term) for term in legacy_terms]

    category_terms = urgency_details.get(category_key)
    if isinstance(category_terms, dict):
        return flatten_category_terms(category_terms)

    return []


def flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    """Flatten component scores into top-level fields for analysis tables."""
    component_scores = record.get("component_scores", {})
    urgency_details = record.get("urgency_details", {})
    return {
        "tweet_id": record.get("tweet_id"),
        "source_row_id": record.get("source_row_id"),
        "role": record.get("role"),
        "disaster_type": record.get("disaster_type"),
        "information_type": record.get("information_type"),
        "prediction_text": record.get("prediction_text"),
        "target_text": record.get("target_text"),
        "reward": float(record.get("reward", 0.0)),
        "relevance": float(component_scores.get("relevance", 0.0)),
        "factuality": float(component_scores.get("factuality", 0.0)),
        "role_coverage": float(component_scores.get("role_coverage", 0.0)),
        "urgency": float(component_scores.get("urgency", 0.0)),
        "roles_scored": "/".join(record.get("roles_scored", [])),
        "unsupported_terms": ", ".join(
            record.get("factuality_details", {}).get("unsupported_terms", [])
        ),
        "urgency_source_terms": ", ".join(
            urgency_terms(
                urgency_details,
                "source_terms",
                "source_terms_by_category",
            )
        ),
        "urgency_candidate_terms": ", ".join(
            urgency_terms(
                urgency_details,
                "candidate_terms",
                "candidate_terms_by_category",
            )
        ),
    }


def quantile(sorted_values: list[float], fraction: float) -> float:
    """Compute a simple linear-interpolated quantile."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]

    position = fraction * (len(sorted_values) - 1)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    weight = position - lower_index
    return (
        sorted_values[lower_index] * (1.0 - weight)
        + sorted_values[upper_index] * weight
    )


def summarize_group(group_name: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize score distributions for a record group."""
    row: dict[str, Any] = {"group": group_name, "num_examples": len(records)}
    for column in SCORE_COLUMNS:
        values = sorted(float(record[column]) for record in records)
        row[f"{column}_mean"] = mean(values) if values else 0.0
        row[f"{column}_median"] = median(values) if values else 0.0
        row[f"{column}_min"] = values[0] if values else 0.0
        row[f"{column}_p25"] = quantile(values, 0.25)
        row[f"{column}_p75"] = quantile(values, 0.75)
        row[f"{column}_max"] = values[-1] if values else 0.0
    return row


def build_summary_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build overall, role, and disaster summary rows."""
    rows = [summarize_group("overall", records)]

    role_counts = Counter(record["role"] for record in records)
    for role in sorted(role_counts):
        role_records = [record for record in records if record["role"] == role]
        rows.append(summarize_group(f"role={role}", role_records))

    disaster_counts = Counter(record["disaster_type"] for record in records)
    for disaster_type, count in sorted(
        disaster_counts.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        if count < 5:
            continue
        disaster_records = [
            record for record in records if record["disaster_type"] == disaster_type
        ]
        rows.append(summarize_group(f"disaster={disaster_type}", disaster_records))

    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Write dictionaries to CSV using the union of keys from the first row."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_summary(rows: list[dict[str, Any]], path: Path) -> None:
    """Write a compact Markdown summary with component means."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        file.write("| group | num_examples | reward_mean | relevance_mean | factuality_mean | role_coverage_mean | urgency_mean |\n")
        file.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            file.write(
                f"| {row['group']} | {row['num_examples']} | "
                f"{row['reward_mean']:.6f} | {row['relevance_mean']:.6f} | "
                f"{row['factuality_mean']:.6f} | {row['role_coverage_mean']:.6f} | "
                f"{row['urgency_mean']:.6f} |\n"
            )


def select_examples(
    records: list[dict[str, Any]],
    column: str,
    count: int,
    descending: bool,
) -> list[dict[str, Any]]:
    """Select high or low examples for a score column."""
    return sorted(records, key=lambda record: record[column], reverse=descending)[:count]


def build_example_rows(records: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """Create inspectable high/low examples for reward and each component."""
    rows: list[dict[str, Any]] = []
    for column in SCORE_COLUMNS:
        for label, descending in [("lowest", False), ("highest", True)]:
            for record in select_examples(records, column, count, descending):
                row = {
                    "selection": f"{label}_{column}",
                    "selected_score": record[column],
                }
                row.update(record)
                rows.append(row)
    return rows


def main() -> int:
    """Analyze reward scores and write report tables."""
    args = parse_args()
    raw_records = read_jsonl(args.input)
    records = [flatten_record(record) for record in raw_records]

    summary_rows = build_summary_rows(records)
    example_rows = build_example_rows(records, args.example_count)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(summary_rows, args.output_dir / "reward_distribution_summary.csv")
    write_markdown_summary(summary_rows, args.output_dir / "reward_distribution_summary.md")
    write_csv(example_rows, args.output_dir / "reward_inspection_examples.csv")

    print(f"Analyzed records: {len(records)}")
    print(f"Wrote reward analysis tables to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
