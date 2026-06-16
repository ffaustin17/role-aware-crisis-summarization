from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


RAW_DATA_PATH = Path("data/raw/frecs.csv")
RANDOM_SEED = 42
OTHER_ROLE_LABEL = "Other"

# Original FReCS columns
ORIGINAL_TWEET_ID_COLUMN = "Tweet ID"
TWEET_TEXT_COLUMN = "Tweet_Text"
INFORMATION_SOURCE_COLUMN = "Information_Source"
INFORMATION_TYPE_COLUMN = "Information_Type"
INFORMATIVENESS_COLUMN = "Informativeness"
FIRST_RESPONDER_COLUMN = "First_Responder"
DISASTER_TYPE_COLUMN = "Disaster_Type"
SECONDARY_ANNOTATION_COLUMN = "Secondary_Annotation"

NULL_EXPORT_COLUMNS = [
    "Unnamed: 8",
    "Unnamed: 9",
    "Unnamed: 10",
    "Unnamed: 11",
    "Unnamed: 12",
]

REQUIRED_COLUMNS = [
    ORIGINAL_TWEET_ID_COLUMN,
    TWEET_TEXT_COLUMN,
    INFORMATION_SOURCE_COLUMN,
    INFORMATION_TYPE_COLUMN,
    INFORMATIVENESS_COLUMN,
    FIRST_RESPONDER_COLUMN,
    DISASTER_TYPE_COLUMN,
    SECONDARY_ANNOTATION_COLUMN,
    *NULL_EXPORT_COLUMNS,
]

RENAME_MAP = {
    TWEET_TEXT_COLUMN: "tweet_text",
    INFORMATION_SOURCE_COLUMN: "information_source",
    INFORMATION_TYPE_COLUMN: "information_type",
    INFORMATIVENESS_COLUMN: "informativeness",
    FIRST_RESPONDER_COLUMN: "role",
    DISASTER_TYPE_COLUMN: "disaster_type",
    SECONDARY_ANNOTATION_COLUMN: "secondary_annotations",
}

FINAL_COLUMNS = [
    "tweet_id",
    "source_row_id",
    "tweet_text",
    "input_text",
    "information_type",
    "information_source",
    "informativeness",
    "disaster_type",
    "role",
    "roles_array",
    "secondary_annotations",
    "secondary_annotations_array",
    "base_summaries",
    "final_base_summary_text",
    "specific_summaries",
    "final_specific_summary_text",
    "generated_by",
]


def load_raw_dataset(path: Path) -> pd.DataFrame:
    """Load the raw FReCS CSV dataset."""
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find dataset at {path}. "
            "Place your raw FReCS CSV at data/raw/frecs.csv."
        )

    return pd.read_csv(
        path,
        dtype={
            ORIGINAL_TWEET_ID_COLUMN: "string",
            TWEET_TEXT_COLUMN: "string",
            INFORMATION_SOURCE_COLUMN: "string",
            INFORMATION_TYPE_COLUMN: "string",
            INFORMATIVENESS_COLUMN: "string",
            FIRST_RESPONDER_COLUMN: "string",
            DISASTER_TYPE_COLUMN: "string",
            SECONDARY_ANNOTATION_COLUMN: "string",
        },
    )


def require_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    """Raise a clear error if the raw dataset is missing expected columns."""
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def normalize_text(value: Any) -> str:
    """Convert a value to a clean string while safely handling missing values."""
    if pd.isna(value):
        return ""

    return str(value).strip()


def split_slash_label(value: Any) -> list[str]:
    """
    Split slash-separated FReCS labels into an ordered list.

    Examples:
        "Police/EMS" -> ["Police", "EMS"]
        "DCC/USAR/MMU" -> ["DCC", "USAR", "MMU"]
        "Other" -> ["Other"]
    """
    cleaned = normalize_text(value)

    if not cleaned:
        return []

    return [part.strip() for part in cleaned.split("/") if part.strip()]


def initialize_summary_dict(labels: list[str]) -> dict[str, str]:
    """
    Initialize a summary dictionary whose keys match a parsed label array.

    Example:
        ["Police", "EMS"] -> {"Police": "", "EMS": ""}
    """
    return {label: "" for label in labels}


def build_input_text(row: pd.Series) -> str:
    """
    Construct the metadata-light model/generator input.

    This first version uses:
    - all assigned responder roles
    - disaster type
    - tweet text

    Other metadata is preserved in the dataframe but not exposed in this first
    input format.
    """
    roles = ", ".join(row["roles_array"])
    disaster_type = normalize_text(row["disaster_type"])
    tweet_text = normalize_text(row["tweet_text"])

    return (
        f"Responder Roles: {roles}\n"
        f"Disaster Type: {disaster_type}\n"
        f"Tweet: {tweet_text}"
    )


def move_other_rows_to_bottom(
    df: pd.DataFrame,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Put responder-relevant rows first while preserving tweet-level rows.

    Non-Other rows are shuffled reproducibly so early generation batches see a
    varied mix of responder-relevant examples. Other rows are preserved and
    placed after them.
    """
    role_labels = df["role"].apply(normalize_text)
    non_other_df = df[role_labels != OTHER_ROLE_LABEL].copy()
    other_df = df[role_labels == OTHER_ROLE_LABEL].copy()

    non_other_df = non_other_df.sample(frac=1, random_state=random_seed).copy()
    other_df = other_df.sample(frac=1, random_state=random_seed).copy()

    return pd.concat([non_other_df, other_df], ignore_index=True).reset_index(
        drop=True
    )


def other_rows_are_at_bottom(df: pd.DataFrame) -> bool:
    """Return True when no non-Other row appears after an Other row."""
    role_labels = df["role"].apply(normalize_text)
    is_other = role_labels == OTHER_ROLE_LABEL
    seen_other = is_other.cummax()

    return bool((~seen_other | is_other).all())


def summary_keys_match_labels(
    row: pd.Series,
    summary_column: str,
    labels_column: str,
) -> bool:
    """Return True when a summary dict's keys exactly match its label array."""
    summary = row[summary_column]
    labels = row[labels_column]

    if not isinstance(summary, dict) or not isinstance(labels, list):
        return False

    return list(summary.keys()) == labels


def build_schema_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw FReCS dataframe into a cleaned tweet-level schema dataframe.

    This function intentionally does not write files. It returns the transformed
    dataframe so later scripts can decide whether to export CSV, JSONL, Excel,
    HTML, or generated training artifacts.
    """
    require_columns(raw_df, REQUIRED_COLUMNS)

    working_df = raw_df.copy()

    original_row_count = len(working_df)

    # Preserve a traceable 1-based source row number before dropping,
    # deduplicating, or reordering rows.
    working_df.insert(0, "source_row_id", range(1, len(working_df) + 1))

    # Remove source columns that should not be carried forward.
    columns_to_drop = [ORIGINAL_TWEET_ID_COLUMN, *NULL_EXPORT_COLUMNS]
    working_df = working_df.drop(columns=columns_to_drop)

    # Rename retained columns to stable snake_case names.
    working_df = working_df.rename(columns=RENAME_MAP)

    # Normalize the tweet text enough for reliable deduplication.
    # We keep the visible tweet text as the stripped text, but we do not perform
    # aggressive cleaning yet because URLs, hashtags, and mentions may carry
    # crisis-relevant signal.
    working_df["tweet_text"] = working_df["tweet_text"].apply(normalize_text)

    # Remove empty tweet rows before deduplication.
    working_df = working_df[working_df["tweet_text"] != ""].copy()

    # Deduplicate tweets. Keep the first matching row.
    working_df = working_df.drop_duplicates(subset=["tweet_text"], keep="first").copy()

    deduplicated_row_count = len(working_df)
    duplicate_rows_removed = original_row_count - deduplicated_row_count

    # Reorder rows before assigning canonical tweet IDs. This keeps the final
    # IDs clean, sequential, and deterministic.
    working_df = move_other_rows_to_bottom(working_df)

    # Add canonical internal tweet_id after deduplication and final reordering.
    working_df.insert(0, "tweet_id", range(1, len(working_df) + 1))

    # Parse slash-separated role and secondary annotation labels.
    working_df["roles_array"] = working_df["role"].apply(split_slash_label)
    working_df["secondary_annotations_array"] = working_df[
        "secondary_annotations"
    ].apply(split_slash_label)

    # Initialize summary scaffolding columns.
    working_df["base_summaries"] = working_df["roles_array"].apply(
        initialize_summary_dict
    )
    working_df["final_base_summary_text"] = ""
    working_df["specific_summaries"] = working_df[
        "secondary_annotations_array"
    ].apply(initialize_summary_dict)
    working_df["final_specific_summary_text"] = ""
    working_df["generated_by"] = ""

    # Build the model/generator input text.
    working_df["input_text"] = working_df.apply(build_input_text, axis=1)

    # Reorder columns into the intended schema.
    working_df = working_df[FINAL_COLUMNS].copy()

    # Store a small amount of debug metadata on the dataframe itself.
    # This is convenient during this testing phase.
    working_df.attrs["original_row_count"] = original_row_count
    working_df.attrs["deduplicated_row_count"] = deduplicated_row_count
    working_df.attrs["duplicate_rows_removed"] = duplicate_rows_removed
    role_labels = working_df["role"].apply(normalize_text)
    working_df.attrs["non_other_row_count"] = int(
        (role_labels != OTHER_ROLE_LABEL).sum()
    )
    working_df.attrs["other_row_count"] = int((role_labels == OTHER_ROLE_LABEL).sum())

    return working_df


def print_distribution(df: pd.DataFrame, column: str) -> None:
    """Print a simple count distribution for a categorical column."""
    print(f"\n=== {column.upper()} DISTRIBUTION ===")
    distribution = (
        df[column]
        .fillna("[MISSING]")
        .astype(str)
        .value_counts(dropna=False)
        .reset_index()
    )
    distribution.columns = [column, "count"]
    distribution["percentage"] = (distribution["count"] / len(df) * 100).round(2)
    print(distribution.to_string(index=False))


def validate_schema_df(schema_df: pd.DataFrame) -> None:
    """Run sanity checks on the transformed schema dataframe."""
    print("\n=== VALIDATION CHECKS ===")

    checks = {
        "final_columns_match_expected": list(schema_df.columns) == FINAL_COLUMNS,
        "tweet_id_is_unique": schema_df["tweet_id"].is_unique,
        "tweet_id_has_no_nulls": schema_df["tweet_id"].notna().all(),
        "tweet_id_starts_at_1": schema_df["tweet_id"].min() == 1,
        "tweet_id_max_equals_row_count": schema_df["tweet_id"].max() == len(schema_df),
        "source_row_id_is_unique": schema_df["source_row_id"].is_unique,
        "source_row_id_has_no_nulls": schema_df["source_row_id"].notna().all(),
        "tweet_text_has_no_nulls": schema_df["tweet_text"].notna().all(),
        "tweet_text_has_no_empty_strings": (schema_df["tweet_text"] != "").all(),
        "tweet_text_is_unique": schema_df["tweet_text"].is_unique,
        "input_text_has_no_nulls": schema_df["input_text"].notna().all(),
        "other_rows_are_at_bottom": other_rows_are_at_bottom(schema_df),
        "roles_array_has_no_empty_arrays": schema_df["roles_array"]
        .apply(lambda labels: len(labels) > 0)
        .all(),
        "secondary_annotations_array_has_no_empty_arrays": schema_df[
            "secondary_annotations_array"
        ]
        .apply(lambda labels: len(labels) > 0)
        .all(),
        "base_summaries_keys_match_roles_array": schema_df.apply(
            lambda row: summary_keys_match_labels(
                row,
                summary_column="base_summaries",
                labels_column="roles_array",
            ),
            axis=1,
        ).all(),
        "specific_summaries_keys_match_secondary_annotations_array": schema_df.apply(
            lambda row: summary_keys_match_labels(
                row,
                summary_column="specific_summaries",
                labels_column="secondary_annotations_array",
            ),
            axis=1,
        ).all(),
    }

    for name, passed in checks.items():
        status = "PASS" if bool(passed) else "FAIL"
        print(f"{status}: {name}")

    failed = [name for name, passed in checks.items() if not bool(passed)]

    if failed:
        raise ValueError(f"Schema validation failed: {failed}")


def print_schema_preview(schema_df: pd.DataFrame) -> None:
    """Print useful preview sections for manual inspection."""
    print("\n=== PIPELINE SHAPE SUMMARY ===")
    print(f"Original rows: {schema_df.attrs['original_row_count']:,}")
    print(f"Final rows after deduplication: {len(schema_df):,}")
    print(f"Duplicate/empty tweet rows removed: {schema_df.attrs['duplicate_rows_removed']:,}")
    print(f"Final columns: {len(schema_df.columns):,}")

    print("\n=== FINAL COLUMN ORDER ===")
    for index, column in enumerate(schema_df.columns, start=1):
        print(f"{index}. {column}")

    print_ordering_summary(schema_df)

    print_distribution(schema_df, "role")
    print_distribution(schema_df, "disaster_type")
    print_distribution(schema_df, "secondary_annotations")
    print_distribution(schema_df, "information_source")
    print_distribution(schema_df, "information_type")
    print_distribution(schema_df, "informativeness")

    print("\n=== PARSED ROLE ARRAY EXAMPLES ===")
    print(
        schema_df[["role", "roles_array"]]
        .drop_duplicates(subset=["role"])
        .to_string(index=False)
    )

    print("\n=== PARSED SECONDARY ANNOTATION ARRAY EXAMPLES ===")
    print(
        schema_df[["secondary_annotations", "secondary_annotations_array"]]
        .drop_duplicates(subset=["secondary_annotations"])
        .head(20)
        .to_string(index=False)
    )

    print("\n=== SUMMARY PLACEHOLDER EXAMPLES ===")
    print(
        schema_df[
            [
                "tweet_id",
                "role",
                "roles_array",
                "base_summaries",
                "secondary_annotations",
                "specific_summaries",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print("\n=== INPUT TEXT EXAMPLES ===")
    for _, row in schema_df.head(5).iterrows():
        print(f"\n--- tweet_id={row['tweet_id']} | role={row['role']} ---")
        print(row["input_text"])


def print_ordering_summary(schema_df: pd.DataFrame) -> None:
    """Print a compact summary of the non-Other/Other row ordering."""
    role_labels = schema_df["role"].apply(normalize_text)
    is_other = role_labels == OTHER_ROLE_LABEL
    non_other_count = int((~is_other).sum())
    other_count = int(is_other.sum())
    other_indexes = schema_df.index[is_other].tolist()
    first_other_index = other_indexes[0] if other_indexes else None
    first_other_position = (
        first_other_index + 1 if first_other_index is not None else "[none]"
    )

    display_columns = [
        "tweet_id",
        "source_row_id",
        "role",
        "disaster_type",
        "secondary_annotations",
    ]

    print("\n=== ORDERING SUMMARY ===")
    print(f"Non-Other rows: {non_other_count:,}")
    print(f"Other rows: {other_count:,}")
    print(f"First Other row position: {first_other_position}")

    print("\n=== FIRST ROWS AFTER REORDERING ===")
    print(schema_df[display_columns].head(10).to_string(index=False))

    if first_other_index is None:
        return

    boundary_start = max(first_other_index - 5, 0)
    boundary_end = min(first_other_index + 5, len(schema_df))

    print("\n=== ROWS AROUND OTHER BOUNDARY ===")
    print(
        schema_df[display_columns]
        .iloc[boundary_start:boundary_end]
        .to_string(index=False)
    )


def main() -> None:
    raw_df = load_raw_dataset(RAW_DATA_PATH)
    schema_df = build_schema_dataframe(raw_df)

    validate_schema_df(schema_df)
    print_schema_preview(schema_df)

    print("\nPipeline completed successfully. No files were written.")


if __name__ == "__main__":
    main()
