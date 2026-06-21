"""Prepare fixed train/validation/test splits for the T5-small baseline."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from sklearn.model_selection import train_test_split


DEFAULT_INPUT_PATH = Path("data/generated/summaries_prompt_v2_input_v3_first_2000.jsonl")
DEFAULT_OUTPUT_DIR = Path("data/modeling/t5_baseline_v1")
DEFAULT_RANDOM_SEED = 42
TEST_SIZE = 0.10
VALIDATION_SIZE = 0.10


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for baseline split preparation."""
    parser = argparse.ArgumentParser(
        description="Create clean T5-small baseline splits from generated summaries."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing split files in the output directory.",
    )
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
    """Return a stripped string, using an empty string for missing values."""
    if value is None:
        return ""
    return str(value).strip()


def clean_success_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Keep the latest successful generated summary for each tweet_id.

    The generation file is append-only, so failed attempts may appear before a
    later success for the same tweet. Only successful records become examples.
    """
    latest_by_tweet_id: dict[int, dict[str, Any]] = {}

    for record in records:
        if record.get("generation_status") != "success":
            continue

        tweet_id = int(record["tweet_id"])
        input_text = normalize_text(record.get("input_text"))
        target_text = normalize_text(record.get("final_base_summary_text"))

        if not input_text:
            raise ValueError(f"tweet_id={tweet_id} has empty input_text")
        if not target_text:
            raise ValueError(f"tweet_id={tweet_id} has empty final_base_summary_text")

        latest_by_tweet_id[tweet_id] = {
            "tweet_id": tweet_id,
            "source_row_id": record.get("source_row_id"),
            "role": normalize_text(record.get("role")),
            "roles_array": record.get("roles_array", []),
            "disaster_type": normalize_text(record.get("disaster_type")),
            "information_type": normalize_text(record.get("information_type")),
            "informativeness": normalize_text(record.get("informativeness")),
            "information_source": normalize_text(record.get("information_source")),
            "tweet_text": normalize_text(record.get("tweet_text")),
            "input_text": input_text,
            "target_text": target_text,
            "input_text_version": normalize_text(record.get("input_text_version")),
            "prompt_version": normalize_text(record.get("prompt_version")),
            "model": normalize_text(record.get("model")),
        }

    return [latest_by_tweet_id[tweet_id] for tweet_id in sorted(latest_by_tweet_id)]


def split_records(
    records: list[dict[str, Any]],
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Create fixed 80/10/10 splits stratified by exact role label."""
    labels = [record["role"] for record in records]
    train_records, holdout_records = train_test_split(
        records,
        test_size=TEST_SIZE + VALIDATION_SIZE,
        random_state=seed,
        shuffle=True,
        stratify=labels,
    )

    holdout_labels = [record["role"] for record in holdout_records]
    validation_records, test_records = train_test_split(
        holdout_records,
        test_size=TEST_SIZE / (TEST_SIZE + VALIDATION_SIZE),
        random_state=seed,
        shuffle=True,
        stratify=holdout_labels,
    )

    return train_records, validation_records, test_records


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    """Write records as UTF-8 JSONL."""
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_role_distribution(
    split_records_by_name: dict[str, list[dict[str, Any]]],
    path: Path,
) -> None:
    """Write a small CSV table showing role counts by split."""
    roles = sorted(
        {
            record["role"]
            for records in split_records_by_name.values()
            for record in records
        }
    )

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["split", "role", "count"])
        for split_name, records in split_records_by_name.items():
            counts = Counter(record["role"] for record in records)
            for role in roles:
                writer.writerow([split_name, role, counts.get(role, 0)])


def validate_splits(split_records_by_name: dict[str, list[dict[str, Any]]]) -> None:
    """Validate that tweet IDs are unique and do not overlap across splits."""
    all_tweet_ids: list[int] = []

    for split_name, records in split_records_by_name.items():
        tweet_ids = [int(record["tweet_id"]) for record in records]
        if len(tweet_ids) != len(set(tweet_ids)):
            raise ValueError(f"{split_name} split has duplicate tweet_id values")
        if any(not record["input_text"] or not record["target_text"] for record in records):
            raise ValueError(f"{split_name} split has empty input_text or target_text")
        all_tweet_ids.extend(tweet_ids)

    if len(all_tweet_ids) != len(set(all_tweet_ids)):
        raise ValueError("tweet_id values overlap across train/validation/test splits")


def write_metadata(
    split_records_by_name: dict[str, list[dict[str, Any]]],
    input_path: Path,
    output_dir: Path,
    seed: int,
) -> None:
    """Write split metadata for reproducibility."""
    metadata = {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "random_seed": seed,
        "split_strategy": "80/10/10 stratified by exact role",
        "input_field": "input_text",
        "target_field": "final_base_summary_text",
        "record_counts": {
            split_name: len(records)
            for split_name, records in split_records_by_name.items()
        },
        "role_counts": {
            split_name: dict(sorted(Counter(record["role"] for record in records).items()))
            for split_name, records in split_records_by_name.items()
        },
    }

    with (output_dir / "split_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)
        file.write("\n")


def main() -> int:
    """Create clean baseline data splits."""
    args = parse_args()
    output_dir: Path = args.output_dir
    output_paths = [
        output_dir / "train.jsonl",
        output_dir / "validation.jsonl",
        output_dir / "test.jsonl",
        output_dir / "split_metadata.json",
        output_dir / "role_distribution.csv",
    ]

    existing_paths = [path for path in output_paths if path.exists()]
    if existing_paths and not args.overwrite:
        existing_text = "\n".join(str(path) for path in existing_paths)
        raise FileExistsError(
            "Refusing to overwrite existing split files. Use --overwrite for:\n"
            f"{existing_text}"
        )

    records = read_jsonl(args.input)
    clean_records = clean_success_records(records)
    train_records, validation_records, test_records = split_records(
        clean_records,
        seed=args.seed,
    )

    split_records_by_name = {
        "train": train_records,
        "validation": validation_records,
        "test": test_records,
    }
    validate_splits(split_records_by_name)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(train_records, output_dir / "train.jsonl")
    write_jsonl(validation_records, output_dir / "validation.jsonl")
    write_jsonl(test_records, output_dir / "test.jsonl")
    write_role_distribution(split_records_by_name, output_dir / "role_distribution.csv")
    write_metadata(split_records_by_name, args.input, output_dir, args.seed)

    print(f"Clean successful examples: {len(clean_records)}")
    for split_name, split_records_for_name in split_records_by_name.items():
        print(f"{split_name}: {len(split_records_for_name)}")
    print(f"Wrote baseline splits to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
