"""Build a presentation-ready CSV joining summary records with reward scores."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_SUMMARIES_PATH = Path("data/modeling/t5_baseline_v1/test_predictions.jsonl")
DEFAULT_REWARDS_PATH = Path(
    "data/rewards/t5_baseline_v1_reward_scores_tweet_relevance_minicheck.jsonl"
)
DEFAULT_OUTPUT_PATH = Path(
    "reports/tables/baseline_predictions_with_tweet_relevance_minicheck_rewards_report.csv"
)

FIELDNAMES = [
    "original_tweet",
    "input_text",
    "candidate_summary",
    "reference_summary",
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
        description="Build a presentation CSV for summary records and rewards."
    )
    parser.add_argument(
        "--summaries",
        "--predictions",
        dest="summaries",
        type=Path,
        default=DEFAULT_SUMMARIES_PATH,
        help=(
            "JSONL records containing source context and candidate summaries. "
            "--predictions is kept as a backwards-compatible alias."
        ),
    )
    parser.add_argument("--rewards", type=Path, default=DEFAULT_REWARDS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--expected-count", type=int, default=None)
    parser.add_argument("--expected-first-tweet-id", type=int, default=None)
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


def candidate_summary(record: dict[str, Any], reward_record: dict[str, Any]) -> str:
    """Return the scored summary from prediction or teacher-summary records."""
    return normalize_text(
        record.get("prediction_text")
        or record.get("final_base_summary_text")
        or reward_record.get("prediction_text")
    )


def reference_summary(record: dict[str, Any]) -> str:
    """Return the supervised target when the record has a separate candidate."""
    if record.get("target_text"):
        return normalize_text(record.get("target_text"))
    if record.get("prediction_text") and record.get("final_base_summary_text"):
        return normalize_text(record.get("final_base_summary_text"))
    return ""


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
    summary_records: list[dict[str, Any]],
    reward_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Join summary and reward records into report rows."""
    rewards_by_tweet_id = build_reward_lookup(reward_records)
    rows = []
    for summary in summary_records:
        tweet_id = int(summary["tweet_id"])
        if tweet_id not in rewards_by_tweet_id:
            raise ValueError(f"No reward record found for tweet_id={tweet_id}")

        reward = rewards_by_tweet_id[tweet_id]
        input_text = normalize_text(summary.get("input_text") or reward.get("input_text"))
        component_scores = reward.get("component_scores", {})
        relevance_details = reward.get("relevance_details", {})
        row = {
            "original_tweet": normalize_text(summary.get("tweet_text"))
            or extract_tweet_from_input_text(input_text),
            "input_text": input_text,
            "candidate_summary": candidate_summary(summary, reward),
            "reference_summary": reference_summary(summary),
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
            "role": normalize_text(summary.get("role") or reward.get("role")),
            "disaster_type": normalize_text(
                summary.get("disaster_type") or reward.get("disaster_type")
            ),
            "information_type": normalize_text(
                summary.get("information_type") or reward.get("information_type")
            ),
            "tweet_id": tweet_id,
            "source_row_id": summary.get("source_row_id") or reward.get("source_row_id"),
        }
        rows.append(row)

    extra_reward_ids = sorted(set(rewards_by_tweet_id) - {int(row["tweet_id"]) for row in rows})
    if extra_reward_ids:
        raise ValueError(
            f"Reward file has {len(extra_reward_ids)} extra tweet_id values; "
            f"first extra is {extra_reward_ids[0]}"
        )

    return rows


def validate_report_rows(
    rows: list[dict[str, Any]],
    expected_count: int | None,
    expected_first_tweet_id: int | None,
) -> None:
    """Validate row count, column order, and required values."""
    if expected_count is not None and len(rows) != expected_count:
        raise ValueError(f"Expected {expected_count} rows, found {len(rows)}")
    if (
        expected_first_tweet_id is not None
        and rows
        and int(rows[0]["tweet_id"]) != expected_first_tweet_id
    ):
        raise ValueError(
            f"Expected first row tweet_id={expected_first_tweet_id}, "
            f"found {rows[0]['tweet_id']}"
        )
    for row_number, row in enumerate(rows, start=1):
        if list(row.keys()) != FIELDNAMES:
            raise ValueError(f"Unexpected column order on row {row_number}")
        for column in NUMERIC_COLUMNS:
            numeric_value(row[column], column)
        if not normalize_text(row["candidate_summary"]):
            raise ValueError(f"Missing candidate_summary on row {row_number}")


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
    summary_records = read_jsonl(args.summaries)
    reward_records = read_jsonl(args.rewards)
    rows = build_report_rows(summary_records, reward_records)
    validate_report_rows(rows, args.expected_count, args.expected_first_tweet_id)
    write_csv(rows, args.output)
    print(f"Summary records: {len(summary_records)}")
    print(f"Reward records: {len(reward_records)}")
    print(f"Wrote report rows: {len(rows)}")
    print(f"Wrote report CSV to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
