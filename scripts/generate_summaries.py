"""Generate external role-aware summary JSONL batches with runtime input builders."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


DEFAULT_INPUT_PATH = Path("data/processed/frecs_training_schema_v2.jsonl")
DEFAULT_OUTPUT_PATH = Path("data/generated/summaries_v1_test.jsonl")
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_LIMIT = 25
DEFAULT_SEED = 42
DEFAULT_MAX_OUTPUT_TOKENS = 400
DEFAULT_RUN_VERSION = "summary_generation_v1"
OTHER_ROLE_LABEL = "Other"

PROMPT_PATHS = {
    "v1": Path("prompts/base_summaries_generation_v1.txt"),
    "v2": Path("prompts/summary_generation_prompt_v2.txt"),
}

ROLE_LABELS = [
    "EMS",
    "Police",
    "Firefighter",
    "Police/EMS",
    "Firefighter/EMS",
    "Police/Firefighter",
    "Police/Firefighter/EMS",
]

URL_PATTERN = re.compile(r"https?://|www\.", re.IGNORECASE)
USERNAME_PATTERN = re.compile(r"(^|\s)@\w+")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for summary generation experiments."""
    parser = argparse.ArgumentParser(
        description="Generate external JSONL summary outputs from FReCS schema rows."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--prompt",
        type=Path,
        default=None,
        help="Optional prompt path override. Defaults are selected by --prompt-version.",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--input-text-version",
        choices=["v1", "v2", "v3"],
        default="v1",
    )
    parser.add_argument(
        "--prompt-version",
        choices=["v1", "v2"],
        default="v1",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Maximum total rows to generate. Keeps defaults safe.",
    )
    parser.add_argument(
        "--limit-per-label",
        type=int,
        default=None,
        help="Optional balanced cap per exact non-Other role label.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip tweet_id values already present in the output file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Clear existing output before generation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected rows and example API inputs without calling the API.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
    )
    parser.add_argument("--run-version", default=DEFAULT_RUN_VERSION)

    return parser.parse_args()


def normalize_text(value: Any) -> str:
    """Convert a loaded JSON value into a clean string."""
    if value is None:
        return ""

    return str(value).strip()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL records from disk."""
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()

            if not stripped:
                continue

            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc

    return records


def read_prompt(path: Path) -> str:
    """Load the external base instruction prompt."""
    if not path.exists():
        raise FileNotFoundError(f"Prompt file does not exist: {path}")

    return path.read_text(encoding="utf-8").strip()


def label_array_text(
    row: dict[str, Any],
    array_column: str,
    fallback_column: str,
) -> str:
    """Return a readable label list from a parsed array or exact-label fallback."""
    labels = row.get(array_column)

    if isinstance(labels, list) and labels:
        return ", ".join(normalize_text(label) for label in labels)

    return normalize_text(row.get(fallback_column))


def roles_text(row: dict[str, Any]) -> str:
    """Return a readable role list from roles_array or role fallback."""
    return label_array_text(row, "roles_array", "role")


def secondary_annotations_text(row: dict[str, Any]) -> str:
    """Return readable secondary annotations from the parsed annotation array."""
    return label_array_text(
        row,
        "secondary_annotations_array",
        "secondary_annotations",
    )


def build_input_text_v1(row: dict[str, Any]) -> tuple[str, list[str]]:
    """Use tweet text, disaster type, and responder roles."""
    return (
        "\n".join(
            [
                f"Responder Roles: {roles_text(row)}",
                f"Disaster Type: {normalize_text(row.get('disaster_type'))}",
                f"Tweet: {normalize_text(row.get('tweet_text'))}",
            ]
        ),
        ["roles_array", "disaster_type", "tweet_text"],
    )


def build_input_text_v2(row: dict[str, Any]) -> tuple[str, list[str]]:
    """Use v1 fields plus parsed FReCS secondary annotations."""
    input_text, fields = build_input_text_v1(row)
    return (
        "\n".join(
            [
                input_text,
                f"Secondary Annotations: {secondary_annotations_text(row)}",
            ]
        ),
        [*fields, "secondary_annotations_array"],
    )


def build_input_text_v3(row: dict[str, Any]) -> tuple[str, list[str]]:
    """Use v2 fields plus the FReCS information type label."""
    input_text, fields = build_input_text_v2(row)
    return (
        "\n".join(
            [
                input_text,
                f"Information Type: {normalize_text(row.get('information_type'))}",
            ]
        ),
        [*fields, "information_type"],
    )


def build_runtime_input_text(
    row: dict[str, Any],
    input_text_version: str,
) -> tuple[str, list[str]]:
    """Build the experiment-specific model input text at runtime."""
    builders = {
        "v1": build_input_text_v1,
        "v2": build_input_text_v2,
        "v3": build_input_text_v3,
    }

    return builders[input_text_version](row)


def resolve_prompt_path(prompt_version: str, prompt_override: Path | None) -> Path:
    """Return the prompt file path for a prompt version or explicit override."""
    if prompt_override is not None:
        return prompt_override

    return PROMPT_PATHS[prompt_version]


def build_api_input(prompt_text: str, input_text: str) -> str:
    """Separate prompt instructions from constructed dataset input."""
    return (
        "Instruction prompt:\n"
        f"{prompt_text}\n\n"
        "Dataset input:\n"
        f"{input_text}\n\n"
        "Return only the requested JSON."
    )


def select_rows(
    records: list[dict[str, Any]],
    limit: int,
    limit_per_label: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    """Select non-Other rows, optionally balancing by exact role label."""
    rng = random.Random(seed)
    non_other_records = [
        record
        for record in records
        if normalize_text(record.get("role")) != OTHER_ROLE_LABEL
    ]

    if limit_per_label is None:
        shuffled = non_other_records.copy()
        rng.shuffle(shuffled)
        return shuffled[: max(limit, 0)]

    selected = []

    for label in ROLE_LABELS:
        label_rows = [
            record
            for record in non_other_records
            if normalize_text(record.get("role")) == label
        ]
        rng.shuffle(label_rows)
        selected.extend(label_rows[: max(limit_per_label, 0)])

    rng.shuffle(selected)
    return selected[: max(limit, 0)]


def load_completed_tweet_ids(path: Path) -> set[int]:
    """Return tweet IDs already present in an existing output file."""
    if not path.exists():
        return set()

    completed = set()

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()

            if not stripped:
                continue

            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                continue

            tweet_id = record.get("tweet_id")

            if isinstance(tweet_id, int):
                completed.add(tweet_id)

    return completed


def extract_response_text(response: object) -> str:
    """Return output text from a Responses API result across SDK variants."""
    output_text = getattr(response, "output_text", None)

    if isinstance(output_text, str):
        return output_text.strip()

    return ""


def validate_generation(
    raw_response: str,
    roles_array: list[str],
) -> tuple[str, dict[str, Any] | None]:
    """Validate the model response before marking a record successful."""
    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        return f"Response is not valid JSON: {exc}", None

    base_summaries = parsed.get("base_summaries")
    final_summary = parsed.get("final_base_summary_text")

    if "base_summaries" not in parsed:
        return "Response is missing base_summaries.", parsed

    if "final_base_summary_text" not in parsed:
        return "Response is missing final_base_summary_text.", parsed

    if not isinstance(base_summaries, dict):
        return "base_summaries must be a dictionary.", parsed

    if set(base_summaries.keys()) != set(roles_array):
        return "base_summaries keys must exactly match roles_array.", parsed

    for role, note in base_summaries.items():
        if not isinstance(note, str) or not note.strip():
            return f"base_summaries[{role!r}] must be a non-empty string.", parsed

    if not isinstance(final_summary, str) or not final_summary.strip():
        return "final_base_summary_text must be a non-empty string.", parsed

    if len(final_summary.split()) > 60:
        return "final_base_summary_text exceeds the 60-word limit.", parsed

    if URL_PATTERN.search(final_summary):
        return "final_base_summary_text contains a URL.", parsed

    if USERNAME_PATTERN.search(final_summary):
        return "final_base_summary_text contains a username.", parsed

    return "", parsed


def create_client() -> Any:
    """Create the OpenAI client without printing or logging secrets."""
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")

    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY or OPENAI_KEY in the environment.")

    return OpenAI(api_key=api_key)


def generate_one(
    client: Any,
    model: str,
    api_input: str,
    max_output_tokens: int,
) -> str:
    """Call the Responses API once and return raw output text."""
    response = client.responses.create(
        model=model,
        input=api_input,
        temperature=0.2,
        max_output_tokens=max_output_tokens,
    )

    return extract_response_text(response)


def build_output_record(
    row: dict[str, Any],
    prompt_path: Path,
    run_version: str,
    input_text_version: str,
    prompt_version: str,
    model: str,
    input_fields_used: list[str],
    constructed_input_text: str,
    generation_status: str,
    raw_response: str = "",
    error_message: str = "",
    parsed_response: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one traceable generated-output JSONL record."""
    parsed_response = parsed_response or {}

    return {
        "run_version": run_version,
        "input_text_version": input_text_version,
        "prompt_version": prompt_version,
        "model": model,
        "generated_by": model,
        "input_fields_used": input_fields_used,
        "input_text": constructed_input_text,
        "tweet_id": row.get("tweet_id"),
        "source_row_id": row.get("source_row_id"),
        "role": row.get("role"),
        "roles_array": row.get("roles_array"),
        "disaster_type": row.get("disaster_type"),
        "secondary_annotations": row.get("secondary_annotations"),
        "secondary_annotations_array": row.get("secondary_annotations_array"),
        "information_type": row.get("information_type"),
        "informativeness": row.get("informativeness"),
        "information_source": row.get("information_source"),
        "tweet_text": row.get("tweet_text"),
        "prompt_path": str(prompt_path),
        "generation_status": generation_status,
        "base_summaries": parsed_response.get("base_summaries", {}),
        "final_base_summary_text": parsed_response.get("final_base_summary_text", ""),
        "quality_notes": parsed_response.get("quality_notes", ""),
        "raw_response": raw_response,
        "error_message": error_message,
    }


def write_jsonl_record(file: Any, record: dict[str, Any]) -> None:
    """Write one JSONL record and flush immediately."""
    file.write(json.dumps(record, ensure_ascii=False) + "\n")
    file.flush()


def print_dry_run(
    selected_rows: list[dict[str, Any]],
    prompt_text: str,
    output_path: Path,
    input_text_version: str,
) -> None:
    """Print selected rows and example API inputs without API calls."""
    print("=== DRY RUN ===")
    print(f"Selected rows: {len(selected_rows)}")
    print(f"Output would be written to: {output_path}")
    print("\nSelected tweet IDs and roles:")

    for row in selected_rows:
        print(f"- tweet_id={row.get('tweet_id')} | role={row.get('role')}")

    print("\nExample API input blocks:")

    for row in selected_rows[:2]:
        input_text, fields = build_runtime_input_text(row, input_text_version)
        print("\n--- BEGIN API INPUT EXAMPLE ---")
        print(f"Input fields used: {fields}")
        print(build_api_input(prompt_text, input_text))
        print("--- END API INPUT EXAMPLE ---")


def main() -> int:
    """Run the small-batch summary generation pipeline."""
    args = parse_args()
    load_dotenv()
    model = args.model or os.getenv("OPENAI_MODEL", DEFAULT_MODEL)

    records = read_jsonl(args.input)
    prompt_path = resolve_prompt_path(args.prompt_version, args.prompt)
    prompt_text = read_prompt(prompt_path)
    selected_rows = select_rows(
        records,
        limit=args.limit,
        limit_per_label=args.limit_per_label,
        seed=args.seed,
    )

    completed_tweet_ids: set[int] = set()

    if args.resume and not args.overwrite:
        completed_tweet_ids = load_completed_tweet_ids(args.output)

    skipped_due_to_resume = [
        row
        for row in selected_rows
        if isinstance(row.get("tweet_id"), int)
        and row["tweet_id"] in completed_tweet_ids
    ]
    rows_to_process = [
        row
        for row in selected_rows
        if not (
            isinstance(row.get("tweet_id"), int)
            and row["tweet_id"] in completed_tweet_ids
        )
    ]

    if args.dry_run:
        print_dry_run(
            selected_rows,
            prompt_text,
            args.output,
            args.input_text_version,
        )
        print("\n=== SUMMARY ===")
        print(f"number selected: {len(selected_rows)}")
        print(f"number skipped due to resume: {len(skipped_due_to_resume)}")
        print("number success: 0")
        print("number validation_failed: 0")
        print("number failed: 0")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.overwrite and args.output.exists():
        args.output.unlink()

    success_count = 0
    validation_failed_count = 0
    failed_count = 0
    client = create_client()

    with args.output.open("a", encoding="utf-8") as output_file:
        for row in rows_to_process:
            tweet_id = row.get("tweet_id")
            role = row.get("role")
            raw_response = ""
            input_text, input_fields_used = build_runtime_input_text(
                row,
                args.input_text_version,
            )

            try:
                raw_response = generate_one(
                    client,
                    model=model,
                    api_input=build_api_input(prompt_text, input_text),
                    max_output_tokens=args.max_output_tokens,
                )
                error_message, parsed_response = validate_generation(
                    raw_response,
                    roles_array=list(row.get("roles_array", [])),
                )

                if error_message:
                    validation_failed_count += 1
                    status = "validation_failed"
                else:
                    success_count += 1
                    status = "success"

                record = build_output_record(
                    row,
                    prompt_path=prompt_path,
                    run_version=args.run_version,
                    input_text_version=args.input_text_version,
                    prompt_version=args.prompt_version,
                    model=model,
                    input_fields_used=input_fields_used,
                    constructed_input_text=input_text,
                    generation_status=status,
                    raw_response=raw_response,
                    error_message=error_message,
                    parsed_response=parsed_response,
                )

            except Exception as exc:
                failed_count += 1
                status = "failed"
                record = build_output_record(
                    row,
                    prompt_path=prompt_path,
                    run_version=args.run_version,
                    input_text_version=args.input_text_version,
                    prompt_version=args.prompt_version,
                    model=model,
                    input_fields_used=input_fields_used,
                    constructed_input_text=input_text,
                    generation_status=status,
                    raw_response=raw_response,
                    error_message=f"{type(exc).__name__}: {exc}",
                )

            write_jsonl_record(output_file, record)
            print(f"tweet_id={tweet_id} | role={role} | status={status}")

    print("\n=== SUMMARY ===")
    print(f"number selected: {len(selected_rows)}")
    print(f"number skipped due to resume: {len(skipped_due_to_resume)}")
    print(f"number success: {success_count}")
    print(f"number validation_failed: {validation_failed_count}")
    print(f"number failed: {failed_count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
