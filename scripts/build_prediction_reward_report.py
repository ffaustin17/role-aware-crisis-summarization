"""Build a presentation-ready CSV joining T5 predictions with reward scores."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_PREDICTIONS_PATH = Path("data/modeling/t5_baseline_v1/test_predictions.jsonl")
DEFAULT_REWARDS_PATH = Path(
    "data/rewards/t5_baseline_v1_reward_scores_tweet_relevance_minicheck.jsonl"
)
DEFAULT_OUTPUT_PATH = Path(
    "reports/tables/baseline_predictions_with_tweet_relevance_minicheck_rewards_report.csv"
)

FIELDNAMES = [
    "original_tweet",
    "input_text",
    "t5_prediction",
    "synthesized_summary_target",
    "reward",
    "relevance",
    "tweet_relevance",
    "context_relevance",
    "factuality",
    "minicheck_support_probability",
    "minicheck_predicted_label",
    "role_coverage",
    "urgency",
    "role",
    "disaster_type",
    "information_type",
    "tweet_id",
    "source_row_id",
]

NUMERIC_COLUMNS = [
    "reward",
    "relevance",
    "tweet_relevance",
    "context_relevance",
    "factuality",
    "role_coverage",
    "urgency",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for report generation."""
    parser = argparse.ArgumentParser(
        description="Build a presentation CSV for baseline predictions and rewards."
    )
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS_PATH)
    parser.add_argument("--rewards", type=Path, default=DEFAULT_REWARDS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--expected-count", type=int, default=401)
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


def normalize_text(value: Any) -> str:
    """Return a stripped string for CSV output."""
    if value is None:
        return ""
    return str(value).strip()


def extract_tweet_from_input_text(input_text: str) -> str:
    """Extract the source tweet from the runtime input text."""
    for line in input_text.splitlines():
        if line.strip().lower().startswith("tweet:"):
            return line.split(":", maxsplit=1)[1].strip()
    return ""


def join_values(values: list[Any]) -> str:
    """Join sentence-level MiniCheck values for CSV inspection."""
    return "; ".join(normalize_text(value) for value in values)


def minicheck_sentence_values(
    reward_record: dict[str, Any],
    key: str,
) -> str:
    """Return semicolon-separated values from MiniCheck sentence details."""
    sentence_scores = (
        reward_record.get("factuality_details", {}).get("sentence_scores", [])
    )
    return join_values([sentence.get(key, "") for sentence in sentence_scores])


def numeric_value(value: Any, column_name: str) -> float:
    """Convert required numeric values and fail clearly if missing."""
    if value is None or value == "":
        raise ValueError(f"Missing numeric value for {column_name}")
    return float(value)


def build_reward_lookup(
    reward_records: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    """Map tweet_id to reward record and reject duplicates."""
    rewards_by_tweet_id: dict[int, dict[str, Any]] = {}
    for record in reward_records:
        tweet_id = int(record["tweet_id"])
        if tweet_id in rewards_by_tweet_id:
            raise ValueError(f"Duplicate reward record for tweet_id={tweet_id}")
        rewards_by_tweet_id[tweet_id] = record
    return rewards_by_tweet_id


def build_report_rows(
    prediction_records: list[dict[str, Any]],
    reward_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Join prediction and reward records into report rows."""
    rewards_by_tweet_id = build_reward_lookup(reward_records)
    rows = []
    for prediction in prediction_records:
        tweet_id = int(prediction["tweet_id"])
        if tweet_id not in rewards_by_tweet_id:
            raise ValueError(f"No reward record found for tweet_id={tweet_id}")

        reward = rewards_by_tweet_id[tweet_id]
        input_text = normalize_text(prediction.get("input_text"))
        component_scores = reward.get("component_scores", {})
        relevance_details = reward.get("relevance_details", {})
        row = {
            "original_tweet": normalize_text(prediction.get("tweet_text"))
            or extract_tweet_from_input_text(input_text),
            "input_text": input_text,
            "t5_prediction": normalize_text(prediction.get("prediction_text")),
            "synthesized_summary_target": normalize_text(prediction.get("target_text")),
            "reward": numeric_value(reward.get("reward"), "reward"),
            "relevance": numeric_value(component_scores.get("relevance"), "relevance"),
            "tweet_relevance": numeric_value(
                relevance_details.get("tweet_relevance"),
                "tweet_relevance",
            ),
            "context_relevance": numeric_value(
                relevance_details.get("context_relevance"),
                "context_relevance",
            ),
            "factuality": numeric_value(component_scores.get("factuality"), "factuality"),
            "minicheck_support_probability": minicheck_sentence_values(
                reward,
                "support_probability",
            ),
            "minicheck_predicted_label": minicheck_sentence_values(
                reward,
                "predicted_label",
            ),
            "role_coverage": numeric_value(
                component_scores.get("role_coverage"),
                "role_coverage",
            ),
            "urgency": numeric_value(component_scores.get("urgency"), "urgency"),
            "role": normalize_text(prediction.get("role") or reward.get("role")),
            "disaster_type": normalize_text(
                prediction.get("disaster_type") or reward.get("disaster_type")
            ),
            "information_type": normalize_text(
                prediction.get("information_type") or reward.get("information_type")
            ),
            "tweet_id": tweet_id,
            "source_row_id": prediction.get("source_row_id") or reward.get("source_row_id"),
        }
        rows.append(row)

    extra_reward_ids = sorted(set(rewards_by_tweet_id) - {int(row["tweet_id"]) for row in rows})
    if extra_reward_ids:
        raise ValueError(
            f"Reward file has {len(extra_reward_ids)} extra tweet_id values; "
            f"first extra is {extra_reward_ids[0]}"
        )

    return rows


def validate_report_rows(rows: list[dict[str, Any]], expected_count: int | None) -> None:
    """Validate row count, column order, and required values."""
    if expected_count is not None and len(rows) != expected_count:
        raise ValueError(f"Expected {expected_count} rows, found {len(rows)}")
    if rows and int(rows[0]["tweet_id"]) != 3611:
        raise ValueError(f"Expected first row tweet_id=3611, found {rows[0]['tweet_id']}")
    for row_number, row in enumerate(rows, start=1):
        if list(row.keys()) != FIELDNAMES:
            raise ValueError(f"Unexpected column order on row {row_number}")
        for column in NUMERIC_COLUMNS:
            numeric_value(row[column], column)


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write report rows to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    """Build and write the presentation report."""
    args = parse_args()
    prediction_records = read_jsonl(args.predictions)
    reward_records = read_jsonl(args.rewards)
    rows = build_report_rows(prediction_records, reward_records)
    validate_report_rows(rows, args.expected_count)
    write_csv(rows, args.output)
    print(f"Prediction records: {len(prediction_records)}")
    print(f"Reward records: {len(reward_records)}")
    print(f"Wrote report rows: {len(rows)}")
    print(f"Wrote report CSV to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
