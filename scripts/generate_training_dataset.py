from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.generate_training_schema_lib import (
    RAW_DATA_PATH,
    build_schema_dataframe,
    load_raw_dataset,
    validate_schema_df,
)


DEFAULT_OUTPUT_CSV_PATH = Path("data/processed/frecs_training.csv")
DEFAULT_OUTPUT_JSONL_PATH = Path("data/processed/frecs_training.jsonl")


def serialize_value(value: Any) -> Any:
    """
    Convert Python objects into JSON-safe values.

    This matters because some dataframe columns contain lists or dictionaries,
    such as roles_array, secondary_annotations_array, base_summaries,
    and specific_summaries.
    """
    if pd.isna(value) if not isinstance(value, (list, dict)) else False:
        return None

    return value


def write_csv(df: pd.DataFrame, output_path: Path) -> None:
    """
    Write the training schema dataframe to CSV.

    List and dictionary columns are serialized as JSON strings so they can be
    safely stored in a flat CSV file and parsed later if needed.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    export_df = df.copy()

    json_like_columns = [
        "roles_array",
        "secondary_annotations_array",
        "base_summaries",
        "specific_summaries",
    ]

    for column in json_like_columns:
        if column in export_df.columns:
            export_df[column] = export_df[column].apply(
                lambda value: json.dumps(value, ensure_ascii=False)
            )

    export_df.to_csv(output_path, index=False, encoding="utf-8")

    print(f"CSV training schema written to: {output_path}")


def write_jsonl(df: pd.DataFrame, output_path: Path) -> None:
    """
    Write the training schema dataframe to JSONL.

    JSONL is often easier than CSV for ML pipelines because list and dictionary
    columns remain naturally structured.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for record in df.to_dict(orient="records"):
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"JSONL training schema written to: {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for schema artifact generation."""
    parser = argparse.ArgumentParser(
        description="Generate the cleaned FReCS training schema dataset."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=RAW_DATA_PATH,
        help="Path to the raw FReCS CSV file.",
    )

    parser.add_argument(
        "--csv-output",
        type=Path,
        default=DEFAULT_OUTPUT_CSV_PATH,
        help="Path where the generated CSV training schema should be saved.",
    )

    parser.add_argument(
        "--jsonl-output",
        type=Path,
        default=DEFAULT_OUTPUT_JSONL_PATH,
        help="Path where the generated JSONL training schema should be saved.",
    )

    parser.add_argument(
        "--jsonl",
        action="store_true",
        help="Also write a JSONL version of the training schema.",
    )

    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Do not write CSV output. Useful when only JSONL is desired.",
    )

    return parser.parse_args()


def main() -> None:
    """Generate training schema artifacts from the raw FReCS dataset."""
    args = parse_args()

    raw_df = load_raw_dataset(args.input)
    schema_df = build_schema_dataframe(raw_df)

    # Reuse the validation from the pipeline script before writing artifacts.
    validate_schema_df(schema_df)

    if not args.no_csv:
        write_csv(schema_df, args.csv_output)

    if args.jsonl:
        write_jsonl(schema_df, args.jsonl_output)

    print("\nTraining schema generation completed successfully.")


if __name__ == "__main__":
    main()