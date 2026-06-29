"""Build DPO preference pairs from GPT teacher and T5 reward scores."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


DEFAULT_T5_SUMMARIES_PATH = Path("data/modeling/t5_baseline_v2/all_predictions.jsonl")
DEFAULT_T5_REWARDS_PATH = Path(
    "data/rewards/t5_baseline_v2_all_predictions_reward_scores_tweet_relevance_minicheck.jsonl"
)
DEFAULT_GPT_SUMMARIES_PATH = Path("data/generated/gpt4o_initial_summaries_v0203.jsonl")
DEFAULT_GPT_REWARDS_PATH = Path(
    "data/rewards/gpt4o_initial_summaries_v0203_reward_scores_tweet_relevance_minicheck.jsonl"
)
DEFAULT_OUTPUT_PATH = Path(
    "data/preferences/dpo_preferences_t5_v2_vs_gpt4o_reward_v1.jsonl"
)
DEFAULT_SUMMARY_CSV_PATH = Path(
    "reports/tables/dpo_preferences_t5_v2_vs_gpt4o_summary.csv"
)
DEFAULT_BY_ROLE_CSV_PATH = Path(
    "reports/tables/dpo_preferences_t5_v2_vs_gpt4o_by_role.csv"
)
DEFAULT_MARGIN_CSV_PATH = Path(
    "reports/tables/dpo_preferences_t5_v2_vs_gpt4o_margin_distribution.csv"
)
DEFAULT_MIN_REWARD_MARGIN = 0.03

COMPONENTS = ["relevance", "factuality", "role_coverage", "urgency"]
MARGIN_BINS = [
    (0.00, 0.01),
    (0.01, 0.03),
    (0.03, 0.05),
    (0.05, 0.10),
    (0.10, 0.20),
    (0.20, 1.000000001),
]
MODEL_LABELS = {
    "t5": "t5_baseline_v2",
    "gpt": "gpt4o_teacher",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build DPO preference pairs from scored GPT and T5 summaries."
    )
    parser.add_argument("--t5-summaries", type=Path, default=DEFAULT_T5_SUMMARIES_PATH)
    parser.add_argument("--t5-rewards", type=Path, default=DEFAULT_T5_REWARDS_PATH)
    parser.add_argument("--gpt-summaries", type=Path, default=DEFAULT_GPT_SUMMARIES_PATH)
    parser.add_argument("--gpt-rewards", type=Path, default=DEFAULT_GPT_REWARDS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV_PATH)
    parser.add_argument("--by-role-csv", type=Path, default=DEFAULT_BY_ROLE_CSV_PATH)
    parser.add_argument("--margin-csv", type=Path, default=DEFAULT_MARGIN_CSV_PATH)
    parser.add_argument(
        "--min-reward-margin",
        type=float,
        default=DEFAULT_MIN_REWARD_MARGIN,
        help="Skip pairs whose absolute reward difference is below this margin.",
    )
    parser.add_argument(
        "--write-split-files",
        action="store_true",
        help="Also write train/validation/test JSONL files next to --output.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing output files.",
    )
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


def normalize_text(value: Any) -> str:
    """Return a stripped string, using empty string for missing values."""
    if value is None:
        return ""
    return str(value).strip()


def build_lookup(records: list[dict[str, Any]], label: str) -> dict[int, dict[str, Any]]:
    """Map records by tweet_id and reject duplicates."""
    lookup: dict[int, dict[str, Any]] = {}
    for record in records:
        tweet_id = int(record["tweet_id"])
        if tweet_id in lookup:
            raise ValueError(f"Duplicate tweet_id={tweet_id} in {label}")
        lookup[tweet_id] = record
    return lookup


def component_scores(record: dict[str, Any]) -> dict[str, float]:
    """Return required reward component scores as floats."""
    scores = record.get("component_scores")
    if not isinstance(scores, dict):
        raise ValueError(f"Missing component_scores for tweet_id={record.get('tweet_id')}")
    return {component: float(scores[component]) for component in COMPONENTS}


def summary_text(record: dict[str, Any], reward_record: dict[str, Any]) -> str:
    """Return the candidate summary represented by a source/reward pair."""
    text = normalize_text(
        record.get("prediction_text")
        or record.get("final_base_summary_text")
        or reward_record.get("prediction_text")
    )
    if not text:
        raise ValueError(f"Missing summary text for tweet_id={record.get('tweet_id')}")
    return text


def paired_component_deltas(
    chosen_scores: dict[str, float],
    rejected_scores: dict[str, float],
) -> dict[str, float]:
    """Return chosen-minus-rejected component deltas."""
    return {
        f"{component}_delta": chosen_scores[component] - rejected_scores[component]
        for component in COMPONENTS
    }


def build_preference_records(
    t5_summaries: list[dict[str, Any]],
    t5_rewards: list[dict[str, Any]],
    gpt_summaries: list[dict[str, Any]],
    gpt_rewards: list[dict[str, Any]],
    min_reward_margin: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build chosen/rejected preference pairs and construction metadata."""
    t5_summary_by_id = build_lookup(t5_summaries, "T5 summaries")
    t5_reward_by_id = build_lookup(t5_rewards, "T5 rewards")
    gpt_summary_by_id = build_lookup(gpt_summaries, "GPT summaries")
    gpt_reward_by_id = build_lookup(gpt_rewards, "GPT rewards")

    tweet_ids = sorted(t5_summary_by_id)
    expected_sets = {
        "t5_rewards": set(t5_reward_by_id),
        "gpt_summaries": set(gpt_summary_by_id),
        "gpt_rewards": set(gpt_reward_by_id),
    }
    for label, tweet_id_set in expected_sets.items():
        if set(tweet_ids) != tweet_id_set:
            missing = sorted(set(tweet_ids) - tweet_id_set)
            extra = sorted(tweet_id_set - set(tweet_ids))
            raise ValueError(
                f"tweet_id mismatch for {label}; "
                f"missing={missing[:3]}, extra={extra[:3]}"
            )

    preferences: list[dict[str, Any]] = []
    skipped_ties = 0
    for tweet_id in tweet_ids:
        t5_summary = t5_summary_by_id[tweet_id]
        t5_reward = t5_reward_by_id[tweet_id]
        gpt_summary = gpt_summary_by_id[tweet_id]
        gpt_reward = gpt_reward_by_id[tweet_id]

        t5_reward_value = float(t5_reward["reward"])
        gpt_reward_value = float(gpt_reward["reward"])
        absolute_margin = abs(gpt_reward_value - t5_reward_value)
        if absolute_margin < min_reward_margin:
            skipped_ties += 1
            continue

        t5_scores = component_scores(t5_reward)
        gpt_scores = component_scores(gpt_reward)
        if gpt_reward_value > t5_reward_value:
            chosen_summary = gpt_summary
            chosen_reward = gpt_reward
            chosen_scores = gpt_scores
            chosen_model = MODEL_LABELS["gpt"]
            rejected_summary = t5_summary
            rejected_reward = t5_reward
            rejected_scores = t5_scores
            rejected_model = MODEL_LABELS["t5"]
        else:
            chosen_summary = t5_summary
            chosen_reward = t5_reward
            chosen_scores = t5_scores
            chosen_model = MODEL_LABELS["t5"]
            rejected_summary = gpt_summary
            rejected_reward = gpt_reward
            rejected_scores = gpt_scores
            rejected_model = MODEL_LABELS["gpt"]

        prompt = normalize_text(t5_summary.get("input_text") or gpt_summary.get("input_text"))
        if not prompt:
            raise ValueError(f"Missing prompt/input_text for tweet_id={tweet_id}")

        chosen_reward_value = float(chosen_reward["reward"])
        rejected_reward_value = float(rejected_reward["reward"])
        reward_margin = chosen_reward_value - rejected_reward_value
        preferences.append(
            {
                "tweet_id": tweet_id,
                "source_row_id": t5_summary.get("source_row_id")
                or gpt_summary.get("source_row_id"),
                "split": normalize_text(t5_summary.get("split")) or "unknown",
                "role": normalize_text(t5_summary.get("role") or gpt_summary.get("role")),
                "disaster_type": normalize_text(
                    t5_summary.get("disaster_type") or gpt_summary.get("disaster_type")
                ),
                "information_type": normalize_text(
                    t5_summary.get("information_type")
                    or gpt_summary.get("information_type")
                ),
                "prompt": prompt,
                "chosen": summary_text(chosen_summary, chosen_reward),
                "rejected": summary_text(rejected_summary, rejected_reward),
                "chosen_model": chosen_model,
                "rejected_model": rejected_model,
                "chosen_reward": chosen_reward_value,
                "rejected_reward": rejected_reward_value,
                "reward_margin": reward_margin,
                "absolute_reward_margin": absolute_margin,
                "chosen_component_scores": chosen_scores,
                "rejected_component_scores": rejected_scores,
                "component_score_deltas": paired_component_deltas(
                    chosen_scores,
                    rejected_scores,
                ),
                "t5_reward": t5_reward_value,
                "gpt4o_reward": gpt_reward_value,
                "t5_summary": summary_text(t5_summary, t5_reward),
                "gpt4o_summary": summary_text(gpt_summary, gpt_reward),
            }
        )

    metadata = {
        "candidate_count": len(tweet_ids),
        "pair_count": len(preferences),
        "skipped_below_margin": skipped_ties,
        "min_reward_margin": min_reward_margin,
    }
    return preferences, metadata


def ensure_can_write(paths: list[Path], overwrite: bool) -> None:
    """Fail before writing if outputs exist and overwrite is not enabled."""
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        formatted = "\n".join(str(path) for path in existing)
        raise FileExistsError(
            "Output files already exist. Use --overwrite to replace:\n"
            f"{formatted}"
        )


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    """Write records as UTF-8 JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Write dictionaries to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def split_output_paths(output_path: Path) -> dict[str, Path]:
    """Return train/validation/test output paths next to the main JSONL."""
    return {
        split: output_path.with_name(f"{output_path.stem}_{split}{output_path.suffix}")
        for split in ["train", "validation", "test"]
    }


def write_split_files(records: list[dict[str, Any]], output_path: Path) -> list[Path]:
    """Write split-specific preference JSONL files."""
    paths = split_output_paths(output_path)
    records_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        records_by_split[record["split"]].append(record)

    written_paths = []
    for split, path in paths.items():
        write_jsonl(records_by_split.get(split, []), path)
        written_paths.append(path)
    return written_paths


def summarize_preferences(
    records: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build a compact overall preference summary."""
    chosen_counts = Counter(record["chosen_model"] for record in records)
    split_counts = Counter(record["split"] for record in records)
    role_counts = Counter(record["role"] for record in records)
    margins = [float(record["reward_margin"]) for record in records]
    rows = [
        {
            "metric": "candidate_count",
            "value": metadata["candidate_count"],
        },
        {
            "metric": "pair_count",
            "value": metadata["pair_count"],
        },
        {
            "metric": "skipped_below_margin",
            "value": metadata["skipped_below_margin"],
        },
        {
            "metric": "min_reward_margin",
            "value": metadata["min_reward_margin"],
        },
        {
            "metric": "mean_reward_margin",
            "value": mean(margins) if margins else 0.0,
        },
        {
            "metric": "median_reward_margin",
            "value": median(margins) if margins else 0.0,
        },
    ]
    rows.extend(
        {"metric": f"chosen_model_count.{model}", "value": count}
        for model, count in sorted(chosen_counts.items())
    )
    rows.extend(
        {"metric": f"split_count.{split}", "value": count}
        for split, count in sorted(split_counts.items())
    )
    rows.extend(
        {"metric": f"role_count.{role}", "value": count}
        for role, count in sorted(role_counts.items())
    )
    return rows


def by_role_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize preference construction by exact role label."""
    rows = []
    for role in sorted({record["role"] for record in records}):
        role_records = [record for record in records if record["role"] == role]
        chosen_counts = Counter(record["chosen_model"] for record in role_records)
        margins = [float(record["reward_margin"]) for record in role_records]
        row = {
            "role": role,
            "pair_count": len(role_records),
            "gpt4o_teacher_chosen": chosen_counts[MODEL_LABELS["gpt"]],
            "t5_baseline_v2_chosen": chosen_counts[MODEL_LABELS["t5"]],
            "mean_reward_margin": mean(margins) if margins else 0.0,
            "median_reward_margin": median(margins) if margins else 0.0,
        }
        for component in COMPONENTS:
            deltas = [
                float(record["component_score_deltas"][f"{component}_delta"])
                for record in role_records
            ]
            row[f"mean_{component}_delta"] = mean(deltas) if deltas else 0.0
        rows.append(row)
    return rows


def margin_distribution_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build histogram-style reward-margin distribution rows."""
    rows = []
    total = len(records)
    for lower, upper in MARGIN_BINS:
        count = sum(lower <= float(record["reward_margin"]) < upper for record in records)
        rows.append(
            {
                "bin_label": f"[{lower:.2f}, {upper:.2f})",
                "bin_min": lower,
                "bin_max": upper if upper < 1.0 else 1.0,
                "count": count,
                "percentage": count / total if total else 0.0,
            }
        )
    return rows


def validate_preferences(records: list[dict[str, Any]]) -> None:
    """Validate DPO preference invariants."""
    seen_tweet_ids = set()
    for index, record in enumerate(records, start=1):
        tweet_id = int(record["tweet_id"])
        if tweet_id in seen_tweet_ids:
            raise ValueError(f"Duplicate preference tweet_id={tweet_id}")
        seen_tweet_ids.add(tweet_id)
        if not normalize_text(record["prompt"]):
            raise ValueError(f"Missing prompt on preference row {index}")
        if not normalize_text(record["chosen"]):
            raise ValueError(f"Missing chosen summary on preference row {index}")
        if not normalize_text(record["rejected"]):
            raise ValueError(f"Missing rejected summary on preference row {index}")
        if record["chosen"] == record["rejected"]:
            raise ValueError(f"Chosen and rejected summaries match for tweet_id={tweet_id}")
        if float(record["chosen_reward"]) <= float(record["rejected_reward"]):
            raise ValueError(f"Non-positive preference margin for tweet_id={tweet_id}")


def main() -> int:
    """Build preference records and analysis tables."""
    args = parse_args()
    split_paths = list(split_output_paths(args.output).values())
    output_paths = [
        args.output,
        args.summary_csv,
        args.by_role_csv,
        args.margin_csv,
    ]
    if args.write_split_files:
        output_paths.extend(split_paths)
    ensure_can_write(output_paths, args.overwrite)

    preferences, metadata = build_preference_records(
        read_jsonl(args.t5_summaries),
        read_jsonl(args.t5_rewards),
        read_jsonl(args.gpt_summaries),
        read_jsonl(args.gpt_rewards),
        args.min_reward_margin,
    )
    validate_preferences(preferences)

    write_jsonl(preferences, args.output)
    written_split_paths: list[Path] = []
    if args.write_split_files:
        written_split_paths = write_split_files(preferences, args.output)

    write_csv(summarize_preferences(preferences, metadata), args.summary_csv)
    write_csv(by_role_rows(preferences), args.by_role_csv)
    write_csv(margin_distribution_rows(preferences), args.margin_csv)

    print(f"Candidate rows: {metadata['candidate_count']}")
    print(f"Preference pairs: {metadata['pair_count']}")
    print(f"Skipped below margin: {metadata['skipped_below_margin']}")
    print(f"Minimum reward margin: {metadata['min_reward_margin']}")
    print(f"Wrote preferences to: {args.output}")
    for path in written_split_paths:
        print(f"Wrote split preferences to: {path}")
    print(f"Wrote summary CSV to: {args.summary_csv}")
    print(f"Wrote by-role CSV to: {args.by_role_csv}")
    print(f"Wrote margin distribution CSV to: {args.margin_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
