"""Analyze summary reward JSONL datasets into CSV tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any


DEFAULT_INPUT_PATH = Path("data/rewards/t5_baseline_v1_reward_scores.jsonl")
DEFAULT_OUTPUT_ROOT = Path("reports/tables/summary_reward_analysis")

SCORE_FIELDS = [
    "reward",
    "relevance",
    "factuality",
    "role_coverage",
    "urgency",
]

LENGTH_FIELDS = [
    "summary_word_count",
    "summary_char_length",
    "summary_sentence_count",
]

NUMERIC_FIELDS = SCORE_FIELDS + LENGTH_FIELDS

GROUP_FIELDS = {
    "role_label": "role",
    "disaster_type": "disaster_type",
    "information_type": "information_type",
}

METRIC_FIELDNAMES = [
    "group_field",
    "group_value",
    "numeric_field",
    "count",
    "mean",
    "median",
    "std",
    "min",
    "p10",
    "p25",
    "p50",
    "p75",
    "p90",
    "max",
]

DISTRIBUTION_FIELDNAMES = [
    "group_field",
    "group_value",
    "numeric_field",
    "bin_label",
    "bin_min",
    "bin_max",
    "count",
    "percentage",
]

WORD_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9'-]*")
SENTENCE_PATTERN = re.compile(r"[^.!?]+[.!?]?")

SCORE_BINS = [
    (0.0, 0.4),
    (0.4, 0.5),
    (0.5, 0.6),
    (0.6, 0.7),
    (0.7, 0.8),
    (0.8, 1.000000001),
]

WORD_COUNT_BINS = [
    (0, 10),
    (10, 20),
    (20, 30),
    (30, 40),
    (40, 60),
    (60, math.inf),
]

CHAR_LENGTH_BINS = [
    (0, 80),
    (80, 160),
    (160, 240),
    (240, 320),
    (320, 480),
    (480, math.inf),
]

SENTENCE_COUNT_BINS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, math.inf),
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze summary reward JSONL datasets into CSV tables."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to reports/tables/summary_reward_analysis/<input stem>/",
    )
    return parser.parse_args()


def default_output_dir(input_path: Path) -> Path:
    """Return the default output directory for an input reward file."""
    return DEFAULT_OUTPUT_ROOT / input_path.stem


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read non-empty JSONL records from disk."""
    records = []
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


def required_float(value: Any, field_name: str, tweet_id: Any) -> float:
    """Return a required numeric value or fail clearly."""
    if value is None or value == "":
        raise ValueError(f"Missing {field_name} for tweet_id={tweet_id}")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Non-numeric {field_name}={value!r} for tweet_id={tweet_id}"
        ) from exc


def normalize_group_value(value: Any) -> str:
    """Normalize grouping values and preserve exact labels."""
    if value is None:
        return "Unknown"
    normalized = str(value).strip()
    return normalized or "Unknown"


def word_count(text: str) -> int:
    """Count word-like tokens in summary text."""
    return len(WORD_PATTERN.findall(text))


def sentence_count(text: str) -> int:
    """Count sentence-like spans in summary text."""
    spans = [match.group(0).strip() for match in SENTENCE_PATTERN.finditer(text)]
    return len([span for span in spans if span])


def flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    """Flatten reward JSONL records into analysis-ready rows."""
    tweet_id = record.get("tweet_id")
    component_scores = record.get("component_scores")
    if not isinstance(component_scores, dict):
        raise ValueError(f"Missing component_scores for tweet_id={tweet_id}")

    summary_text = str(record.get("prediction_text") or "")
    flattened = {
        "tweet_id": tweet_id,
        "source_row_id": record.get("source_row_id"),
        "role": normalize_group_value(record.get("role")),
        "disaster_type": normalize_group_value(record.get("disaster_type")),
        "information_type": normalize_group_value(record.get("information_type")),
        "reward": required_float(record.get("reward"), "reward", tweet_id),
        "relevance": required_float(
            component_scores.get("relevance"),
            "component_scores.relevance",
            tweet_id,
        ),
        "factuality": required_float(
            component_scores.get("factuality"),
            "component_scores.factuality",
            tweet_id,
        ),
        "role_coverage": required_float(
            component_scores.get("role_coverage"),
            "component_scores.role_coverage",
            tweet_id,
        ),
        "urgency": required_float(
            component_scores.get("urgency"),
            "component_scores.urgency",
            tweet_id,
        ),
        "summary_word_count": word_count(summary_text),
        "summary_char_length": len(summary_text),
        "summary_sentence_count": sentence_count(summary_text),
    }
    return flattened


def quantile(sorted_values: list[float], fraction: float) -> float:
    """Compute a linear-interpolated quantile."""
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


def summarize_values(
    records: list[dict[str, Any]],
    numeric_field: str,
    group_field: str,
    group_value: str,
) -> dict[str, Any]:
    """Summarize one numeric field for a group of records."""
    values = sorted(float(record[numeric_field]) for record in records)
    return {
        "group_field": group_field,
        "group_value": group_value,
        "numeric_field": numeric_field,
        "count": len(values),
        "mean": mean(values) if values else 0.0,
        "median": median(values) if values else 0.0,
        "std": stdev(values) if len(values) > 1 else 0.0,
        "min": values[0] if values else 0.0,
        "p10": quantile(values, 0.10),
        "p25": quantile(values, 0.25),
        "p50": quantile(values, 0.50),
        "p75": quantile(values, 0.75),
        "p90": quantile(values, 0.90),
        "max": values[-1] if values else 0.0,
    }


def build_metric_rows(
    records: list[dict[str, Any]],
    group_field: str,
    source_field: str | None,
) -> list[dict[str, Any]]:
    """Build metric rows overall or grouped by an exact categorical field."""
    rows = []
    if source_field is None:
        for numeric_field in NUMERIC_FIELDS:
            rows.append(
                summarize_values(records, numeric_field, group_field, "overall")
            )
        return rows

    grouped_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped_records[record[source_field]].append(record)

    for group_value in sorted(grouped_records):
        for numeric_field in NUMERIC_FIELDS:
            rows.append(
                summarize_values(
                    grouped_records[group_value],
                    numeric_field,
                    group_field,
                    group_value,
                )
            )
    return rows


def bins_for_field(numeric_field: str) -> list[tuple[float, float]]:
    """Return histogram bins for a numeric field."""
    if numeric_field in SCORE_FIELDS:
        return SCORE_BINS
    if numeric_field == "summary_word_count":
        return WORD_COUNT_BINS
    if numeric_field == "summary_char_length":
        return CHAR_LENGTH_BINS
    if numeric_field == "summary_sentence_count":
        return SENTENCE_COUNT_BINS
    raise ValueError(f"No bins configured for {numeric_field}")


def format_bin_label(lower: float, upper: float) -> str:
    """Format a compact bin label."""
    if math.isinf(upper):
        return f"{lower:g}+"
    return f"{lower:g}-{upper:g}"


def build_distribution_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build overall histogram rows for all numeric fields."""
    rows = []
    for numeric_field in NUMERIC_FIELDS:
        values = [float(record[numeric_field]) for record in records]
        total = len(values)
        for lower, upper in bins_for_field(numeric_field):
            count = sum(lower <= value < upper for value in values)
            rows.append(
                {
                    "group_field": "overall",
                    "group_value": "overall",
                    "numeric_field": numeric_field,
                    "bin_label": format_bin_label(lower, upper),
                    "bin_min": lower,
                    "bin_max": "inf" if math.isinf(upper) else upper,
                    "count": count,
                    "percentage": count / total if total else 0.0,
                }
            )
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path, fieldnames: list[str]) -> None:
    """Write rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def validate_known_means(records: list[dict[str, Any]], input_path: Path) -> None:
    """Validate known reward means for the current committed reward fixtures."""
    known_means = {
        "t5_baseline_v1_reward_scores.jsonl": {
            "reward": 0.6009913209590798,
            "relevance": 0.7902606438817527,
            "factuality": 0.5273925686030293,
            "role_coverage": 0.516209476309227,
            "urgency": 0.44655029093931836,
        },
        "t5_baseline_v1_reward_scores_tweet_relevance_minicheck.jsonl": {
            "reward": 0.5626723172892882,
            "relevance": 0.7149491435111943,
            "factuality": 0.4795526544426445,
            "role_coverage": 0.516209476309227,
            "urgency": 0.44655029093931836,
        },
    }
    expected = known_means.get(input_path.name)
    if expected is None:
        return
    for field_name, expected_mean in expected.items():
        actual_mean = mean(float(record[field_name]) for record in records)
        if not math.isclose(actual_mean, expected_mean, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"{field_name} mean mismatch for {input_path.name}: "
                f"expected {expected_mean}, found {actual_mean}"
            )


def main() -> int:
    """Analyze a summary reward JSONL file."""
    args = parse_args()
    output_dir = args.output_dir or default_output_dir(args.input)
    raw_records = read_jsonl(args.input)
    records = [flatten_record(record) for record in raw_records]
    validate_known_means(records, args.input)

    overall_rows = build_metric_rows(records, "overall", None)
    by_role_rows = build_metric_rows(records, "role_label", GROUP_FIELDS["role_label"])
    by_disaster_rows = build_metric_rows(
        records,
        "disaster_type",
        GROUP_FIELDS["disaster_type"],
    )
    by_information_rows = build_metric_rows(
        records,
        "information_type",
        GROUP_FIELDS["information_type"],
    )
    distribution_rows = build_distribution_rows(records)

    write_csv(overall_rows, output_dir / "overall_metrics.csv", METRIC_FIELDNAMES)
    write_csv(by_role_rows, output_dir / "by_role_label.csv", METRIC_FIELDNAMES)
    write_csv(
        by_disaster_rows,
        output_dir / "by_disaster_type.csv",
        METRIC_FIELDNAMES,
    )
    write_csv(
        by_information_rows,
        output_dir / "by_information_type.csv",
        METRIC_FIELDNAMES,
    )
    write_csv(
        distribution_rows,
        output_dir / "distributions.csv",
        DISTRIBUTION_FIELDNAMES,
    )

    print(f"Analyzed records: {len(records)}")
    print(f"Wrote analysis CSVs to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
