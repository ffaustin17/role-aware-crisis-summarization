from pathlib import Path
from typing import Iterable

import pandas as pd


RAW_DATA_PATH = Path("data/raw/frecs.csv")
REPORT_TABLES_DIR = Path("reports/tables")
INVENTORY_OUTPUT_PATH = REPORT_TABLES_DIR / "frecs_dataset_inventory.xlsx"

# Dataset-specific column names.
TWEET_ID_COLUMN = "Tweet ID"
TWEET_TEXT_COLUMN = "Tweet_Text"
FIRST_RESPONDER_COLUMN = "First_Responder"
DISASTER_TYPE_COLUMN = "Disaster_Type"
SECONDARY_ANNOTATION_COLUMN = "Secondary_Annotation"
INFORMATION_SOURCE_COLUMN = "Information_Source"
INFORMATION_TYPE_COLUMN = "Information_Type"
INFORMATIVENESS_COLUMN = "Informativeness"

BASE_RESPONDER_ROLES = ["EMS", "Police", "Firefighter"]

# Exact labels expected in the First_Responder field.
def get_exact_labels_from_distribution(
    distribution_df: pd.DataFrame,
    label_column: str,
) -> list[str]:
    """Return exact labels in distribution order."""
    return distribution_df[label_column].astype(str).tolist()


def load_dataset(path: Path) -> pd.DataFrame:
    """Load the raw FReCS CSV dataset."""
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find dataset at {path}. "
            "Place the CSV file at data/raw/frecs.csv."
        )

    return pd.read_csv(
        path,
        dtype={
            TWEET_ID_COLUMN: str,
            TWEET_TEXT_COLUMN: str,
            INFORMATION_SOURCE_COLUMN: str,
            INFORMATION_TYPE_COLUMN: str,
            INFORMATIVENESS_COLUMN: str,
            FIRST_RESPONDER_COLUMN: str,
            DISASTER_TYPE_COLUMN: str,
            SECONDARY_ANNOTATION_COLUMN: str,
        },
    )


def require_columns(df: pd.DataFrame, required_columns: Iterable[str]) -> None:
    """Fail early if the expected dataset columns are missing."""
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(
            "The dataset is missing required columns: "
            + ", ".join(missing_columns)
        )


def normalize_label(value: object) -> str:
    """Convert a label value into a clean string."""
    if pd.isna(value):
        return ""

    return str(value).strip()


def split_responder_label(label: object) -> set[str]:
    """
    Convert a First_Responder label into its component roles.

    Examples:
        "EMS" -> {"EMS"}
        "EMS/Police" -> {"EMS", "Police"}
        "EMS/Police/Firefighter" -> {"EMS", "Police", "Firefighter"}
        "Other" -> {"Other"}
    """
    normalized = normalize_label(label)

    if not normalized:
        return set()

    return {part.strip() for part in normalized.split("/") if part.strip()}


def label_contains_base_role(label: object, base_role: str) -> bool:
    """Return True if a First_Responder label contains a base role."""
    return base_role in split_responder_label(label)


def build_column_inventory(df: pd.DataFrame) -> pd.DataFrame:
    """Build a compact inventory of all columns."""
    rows = []

    for col in df.columns:
        non_null_count = int(df[col].notna().sum())
        null_count = int(df[col].isna().sum())

        rows.append(
            {
                "column_name": col,
                "dtype": str(df[col].dtype),
                "non_null_count": non_null_count,
                "null_count": null_count,
                "unique_count": int(df[col].nunique(dropna=True)),
                "is_fully_null": null_count == len(df),
            }
        )

    return pd.DataFrame(rows)


def build_exact_distribution(
    df: pd.DataFrame,
    column: str,
    label_name: str,
) -> pd.DataFrame:
    """Build a count and percentage distribution for a categorical column."""
    counts = (
        df[column]
        .fillna("[MISSING]")
        .astype(str)
        .str.strip()
        .value_counts(dropna=False)
        .reset_index()
    )

    counts.columns = [label_name, "count"]
    counts["percentage"] = (counts["count"] / len(df) * 100).round(2)

    return counts


def build_role_membership_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count how many rows are related to each base responder role.

    EMS-related includes:
        EMS, EMS/Police, EMS/Firefighter, EMS/Police/Firefighter

    Police-related includes:
        Police, EMS/Police, Police/Firefighter, EMS/Police/Firefighter

    Firefighter-related includes:
        Firefighter, EMS/Firefighter, Police/Firefighter, EMS/Police/Firefighter
    """
    rows = []

    for role in BASE_RESPONDER_ROLES:
        mask = df[FIRST_RESPONDER_COLUMN].apply(
            lambda label: label_contains_base_role(label, role)
        )

        count = int(mask.sum())

        rows.append(
            {
                "base_role": role,
                "related_row_count": count,
                "percentage_of_dataset": round(count / len(df) * 100, 2),
            }
        )

    return pd.DataFrame(rows)


def build_overview(df: pd.DataFrame) -> pd.DataFrame:
    """Build a small one-sheet dataset overview."""
    unique_tweets = df[TWEET_TEXT_COLUMN].nunique(dropna=True)
    unique_tweet_ids = df[TWEET_ID_COLUMN].nunique(dropna=True)

    return pd.DataFrame(
        [
            {"metric": "total_rows", "value": len(df)},
            {"metric": "total_columns", "value": len(df.columns)},
            {"metric": "unique_tweet_texts", "value": unique_tweets},
            {"metric": "unique_tweet_ids", "value": unique_tweet_ids},
            {
                "metric": "duplicate_tweet_text_rows",
                "value": len(df) - unique_tweets,
            },
            {
                "metric": "duplicate_tweet_id_rows",
                "value": len(df) - unique_tweet_ids,
            },
        ]
    )


def filter_by_exact_responder_label(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Return rows whose First_Responder label exactly matches the given label."""
    return df[df[FIRST_RESPONDER_COLUMN].astype(str).str.strip() == label].copy()


def filter_by_base_responder_role(df: pd.DataFrame, base_role: str) -> pd.DataFrame:
    """Return rows related to a base role: EMS, Police, or Firefighter."""
    if base_role not in BASE_RESPONDER_ROLES:
        raise ValueError(
            f"Unknown base role: {base_role}. "
            f"Expected one of: {BASE_RESPONDER_ROLES}"
        )

    mask = df[FIRST_RESPONDER_COLUMN].apply(
        lambda label: label_contains_base_role(label, base_role)
    )

    return df[mask].copy()


def sample_rows(
    df: pd.DataFrame,
    n: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Return up to n random rows from a dataframe."""
    if df.empty:
        return df.copy()

    sample_size = min(n, len(df))

    return df.sample(n=sample_size, random_state=random_state).copy()


def select_sample_display_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep samples readable in the CLI.

    Internally, filtered dataframes preserve all columns.
    For display, we only show the most useful human-inspection columns.
    """
    display_columns = [
        TWEET_TEXT_COLUMN,
        DISASTER_TYPE_COLUMN,
        FIRST_RESPONDER_COLUMN,
        SECONDARY_ANNOTATION_COLUMN,
    ]

    existing_display_columns = [col for col in display_columns if col in df.columns]

    return df[existing_display_columns].copy()


def write_inventory_xlsx(
    output_path: Path,
    overview_df: pd.DataFrame,
    column_inventory_df: pd.DataFrame,
    role_exact_distribution_df: pd.DataFrame,
    role_membership_distribution_df: pd.DataFrame,
    disaster_type_distribution_df: pd.DataFrame,
    secondary_annotation_distribution_df: pd.DataFrame,
) -> None:
    """Write a compact multi-sheet Excel inventory."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path) as writer:
        overview_df.to_excel(writer, sheet_name="overview", index=False)
        column_inventory_df.to_excel(writer, sheet_name="columns", index=False)
        role_exact_distribution_df.to_excel(
            writer,
            sheet_name="first_responder_exact",
            index=False,
        )
        role_membership_distribution_df.to_excel(
            writer,
            sheet_name="role_membership",
            index=False,
        )
        disaster_type_distribution_df.to_excel(
            writer,
            sheet_name="disaster_types",
            index=False,
        )
        secondary_annotation_distribution_df.to_excel(
            writer,
            sheet_name="secondary_annotations",
            index=False,
        )


def format_cli_table(df: pd.DataFrame) -> str:
    """
    Format a dataframe for readable CLI output.

    Text-like values are left-aligned.
    Numeric values are right-aligned.
    """
    if df.empty:
        return "[empty dataframe]"

    display_df = df.copy()

    # Convert values to strings while preserving missing readability.
    for col in display_df.columns:
        display_df[col] = display_df[col].apply(
            lambda value: "" if pd.isna(value) else str(value)
        )

    column_widths: dict[str, int] = {}

    for col in display_df.columns:
        max_value_width = display_df[col].map(len).max()
        column_widths[col] = max(len(str(col)), int(max_value_width))

    lines = []

    header_parts = [
        str(col).ljust(column_widths[col])
        for col in display_df.columns
    ]
    lines.append("  ".join(header_parts))

    separator_parts = [
        "-" * column_widths[col]
        for col in display_df.columns
    ]
    lines.append("  ".join(separator_parts))

    original_dtypes = df.dtypes

    for _, row in display_df.iterrows():
        row_parts = []

        for col in display_df.columns:
            value = row[col]

            if pd.api.types.is_numeric_dtype(original_dtypes[col]):
                row_parts.append(value.rjust(column_widths[col]))
            else:
                row_parts.append(value.ljust(column_widths[col]))

        lines.append("  ".join(row_parts))

    return "\n".join(lines)

def print_distribution(title: str, distribution_df: pd.DataFrame) -> None:
    """Print a distribution dataframe with a section title."""
    print(f"\n=== {title} ===")
    print(format_cli_table(distribution_df))


def print_role_samples(
        df: pd.DataFrame, 
        role_exact_distribution_df: pd.DataFrame,   
        n: int = 3
    ) -> None:
    """
    Print random tweet samples for exact role labels and base role memberships.

    Exact role labels are read from the actual dataset distribution instead of
    hardcoded assumptions. This matters because labels may be ordered as
    Police/EMS rather than EMS/Police.
    """
    print("\n=== RANDOM SAMPLES BY EXACT FIRST_RESPONDER LABEL ===")

    exact_labels = get_exact_labels_from_distribution(
        role_exact_distribution_df,
        label_column="first_responder_label",
    )

    for label in exact_labels:
        label_df = filter_by_exact_responder_label(df, label)
        sampled = sample_rows(label_df, n=n)

        print(f"\n--- Exact label: {label} | rows: {len(label_df):,} ---")

        if sampled.empty:
            print("No rows found.")
            continue

        print(format_cli_table(select_sample_display_columns(sampled)))

    print("\n=== RANDOM SAMPLES BY BASE ROLE MEMBERSHIP ===")

    for base_role in BASE_RESPONDER_ROLES:
        role_df = filter_by_base_responder_role(df, base_role)
        sampled = sample_rows(role_df, n=n)

        print(f"\n--- {base_role}-related rows: {len(role_df):,} ---")

        if sampled.empty:
            print("No rows found.")
            continue

        print(format_cli_table(select_sample_display_columns(sampled)))


def print_cli_report(
    df: pd.DataFrame,
    overview_df: pd.DataFrame,
    column_inventory_df: pd.DataFrame,
    role_exact_distribution_df: pd.DataFrame,
    role_membership_distribution_df: pd.DataFrame,
    disaster_type_distribution_df: pd.DataFrame,
    secondary_annotation_distribution_df: pd.DataFrame,
    sample_n: int = 3,
) -> None:
    """Print the main exploration report to the CLI."""
    print("\n=== BASIC DATASET INFO ===")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")

    print("\n=== COLUMN NAMES ===")
    for i, col in enumerate(df.columns, start=1):
        print(f"{i}. {col}")

    print("\n=== OVERVIEW ===")
    print(format_cli_table(overview_df))

    print("\n=== COLUMN INVENTORY ===")
    print(format_cli_table(column_inventory_df))

    print_distribution(
        "FIRST_RESPONDER EXACT LABEL DISTRIBUTION",
        role_exact_distribution_df,
    )

    print_role_label_notes(
        role_exact_distribution_df=role_exact_distribution_df,   
    )
    
    print_distribution(
        "BASE ROLE MEMBERSHIP DISTRIBUTION",
        role_membership_distribution_df,
    )

    print_distribution(
        "DISASTER_TYPE DISTRIBUTION",
        disaster_type_distribution_df,
    )

    print_distribution(
        "SECONDARY_ANNOTATION DISTRIBUTION",
        secondary_annotation_distribution_df,
    )

    print_role_samples(
        df=df, 
        role_exact_distribution_df=role_exact_distribution_df,
        n=sample_n,
    )


def print_role_label_notes(role_exact_distribution_df: pd.DataFrame) -> None:
    """Print notes about the exact First_Responder labels found in the dataset"""
    exact_labels = set(
        role_exact_distribution_df["first_responder_label"]
        .astype(str)
        .tolist()
    )

    print("\n=== FIRST_RESPONDER LABEL NOTES ===")

    for label in sorted(exact_labels):
        components = split_responder_label(label)
        print(f"{label} -> {sorted(components)}")

def main() -> None:
    df = load_dataset(RAW_DATA_PATH)

    require_columns(
        df,
        required_columns=[
            TWEET_ID_COLUMN,
            TWEET_TEXT_COLUMN,
            FIRST_RESPONDER_COLUMN,
            DISASTER_TYPE_COLUMN,
            SECONDARY_ANNOTATION_COLUMN,
        ],
    )

    overview_df = build_overview(df)
    column_inventory_df = build_column_inventory(df)

    role_exact_distribution_df = build_exact_distribution(
        df,
        column=FIRST_RESPONDER_COLUMN,
        label_name="first_responder_label",
    )

    role_membership_distribution_df = build_role_membership_distribution(df)

    disaster_type_distribution_df = build_exact_distribution(
        df,
        column=DISASTER_TYPE_COLUMN,
        label_name="disaster_type",
    )

    secondary_annotation_distribution_df = build_exact_distribution(
        df,
        column=SECONDARY_ANNOTATION_COLUMN,
        label_name="secondary_annotation",
    )

    print_cli_report(
        df=df,
        overview_df=overview_df,
        column_inventory_df=column_inventory_df,
        role_exact_distribution_df=role_exact_distribution_df,
        role_membership_distribution_df=role_membership_distribution_df,
        disaster_type_distribution_df=disaster_type_distribution_df,
        secondary_annotation_distribution_df=secondary_annotation_distribution_df,
        sample_n=3,
    )

    write_inventory_xlsx(
        output_path=INVENTORY_OUTPUT_PATH,
        overview_df=overview_df,
        column_inventory_df=column_inventory_df,
        role_exact_distribution_df=role_exact_distribution_df,
        role_membership_distribution_df=role_membership_distribution_df,
        disaster_type_distribution_df=disaster_type_distribution_df,
        secondary_annotation_distribution_df=secondary_annotation_distribution_df,
    )

    print(f"\nDataset inventory saved to {INVENTORY_OUTPUT_PATH}")


if __name__ == "__main__":
    main()

