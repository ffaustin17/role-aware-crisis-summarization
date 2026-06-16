from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


DEFAULT_INPUT_PATH = Path("data/processed/frecs_training.jsonl")
DEFAULT_OUTPUT_PATH = Path("data/generated/base_summaries_v1_test.jsonl")
DEFAULT_PROMPT_PATH = Path("prompts/base_summaries_generation_v1.txt")
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_LIMIT_PER_LABEL = 2
DEFAULT_SEED = 42
DEFAULT_MAX_OUTPUT_TOKENS = 400
OTHER_ROLE_LABEL = "Other"
PROMPT_VERSION = "base_summary_prompt_v1"

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
    """Parse command-line arguments for small-batch summary generation."""
    parser = argparse.ArgumentParser(
        description="Generate a small external JSONL batch of base summaries."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to the processed FReCS training JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to the generated summary JSONL output file.",
    )
    parser.add_argument(
        "--prompt",
        type=Path,
        default=DEFAULT_PROMPT_PATH,
        help="Path to the external instruction prompt file.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional OpenAI model override. Defaults to OPENAI_MODEL or gpt-4o-mini.",
    )
    parser.add_argument(
        "--limit-per-label",
        type=int,
        default=DEFAULT_LIMIT_PER_LABEL,
        help="Maximum rows to sample per exact non-Other role label.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for reproducible sampling.",
    )
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
        help="Print selected rows and example prompts without calling the API.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
        help="Maximum output tokens for each Responses API call.",
    )

    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dictionaries."""
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()

            if not stripped:
                continue

            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc

            records.append(record)

    return records


def read_prompt(path: Path) -> str:
    """Load the external instruction prompt."""
    if not path.exists():
        raise FileNotFoundError(f"Prompt file does not exist: {path}")

    return path.read_text(encoding="utf-8").strip()


def load_completed_tweet_ids(path: Path) -> set[int]:
    """Return tweet IDs already present in an existing output JSONL file."""
    if not path.exists():
        return set()

    completed_tweet_ids = set()

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
                completed_tweet_ids.add(tweet_id)

    return completed_tweet_ids


def select_balanced_rows(
    records: list[dict[str, Any]],
    limit_per_label: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Sample up to limit_per_label rows from each exact non-Other role label."""
    if limit_per_label <= 0:
        return []

    selected = []
    rng = random.Random(seed)

    for label in ROLE_LABELS:
        label_rows = [
            record
            for record in records
            if str(record.get("role", "")).strip() == label
        ]
        rng.shuffle(label_rows)
        selected.extend(label_rows[:limit_per_label])

    return selected


def build_api_input(prompt_text: str, input_text: str) -> str:
    """Combine the external instruction prompt and canonical dataset input."""
    return (
        "Instruction prompt:\n"
        f"{prompt_text}\n\n"
        "Dataset input:\n"
        f"{input_text}\n\n"
        "Return only the requested JSON."
    )


def extract_response_text(response: object) -> str:
    """Return output text from a Responses API result across SDK variants."""
    output_text = getattr(response, "output_text", None)

    if isinstance(output_text, str):
        return output_text.strip()

    return ""


def word_count(text: str) -> int:
    """Count rough whitespace-delimited words in generated text."""
    return len(text.split())


def validate_generation(
    raw_response: str,
    roles_array: list[str],
) -> tuple[str, dict[str, Any] | None]:
    """Validate raw model JSON and return an error message plus parsed object."""
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

    if word_count(final_summary) > 60:
        return "final_base_summary_text exceeds the 60-word limit.", parsed

    if URL_PATTERN.search(final_summary):
        return "final_base_summary_text contains a URL.", parsed

    if USERNAME_PATTERN.search(final_summary):
        return "final_base_summary_text contains a username.", parsed

    return "", parsed


def build_output_record(
    row: dict[str, Any],
    prompt_path: Path,
    model: str,
    generation_status: str,
    raw_response: str = "",
    error_message: str = "",
    parsed_response: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one generated-output JSONL record."""
    parsed_response = parsed_response or {}

    return {
        "tweet_id": row.get("tweet_id"),
        "source_row_id": row.get("source_row_id"),
        "role": row.get("role"),
        "roles_array": row.get("roles_array"),
        "disaster_type": row.get("disaster_type"),
        "tweet_text": row.get("tweet_text"),
        "input_text": row.get("input_text"),
        "prompt_path": str(prompt_path),
        "prompt_version": PROMPT_VERSION,
        "generated_by": model,
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


def create_client() -> Any:
    """Create an OpenAI client from environment variables."""
    from openai import OpenAI

    load_dotenv()

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
    """Call the OpenAI Responses API once and return raw output text."""
    response = client.responses.create(
        model=model,
        input=api_input,
        temperature=0.2,
        max_output_tokens=max_output_tokens,
    )

    return extract_response_text(response)


def print_dry_run(
    selected_rows: list[dict[str, Any]],
    prompt_text: str,
    output_path: Path,
) -> None:
    """Print selected rows and example API inputs without writing output."""
    print("=== DRY RUN ===")
    print(f"Selected rows: {len(selected_rows)}")
    print(f"Output would be written to: {output_path}")
    print("\nSelected tweet IDs and roles:")

    for row in selected_rows:
        print(f"- tweet_id={row.get('tweet_id')} | role={row.get('role')}")

    print("\nExample API input blocks:")

    for row in selected_rows[:2]:
        print("\n--- BEGIN API INPUT EXAMPLE ---")
        print(build_api_input(prompt_text, str(row.get("input_text", ""))))
        print("--- END API INPUT EXAMPLE ---")


def main() -> int:
    args = parse_args()
    load_dotenv()
    model = args.model or os.getenv("OPENAI_MODEL", DEFAULT_MODEL)

    records = read_jsonl(args.input)
    prompt_text = read_prompt(args.prompt)
    selected_rows = select_balanced_rows(
        records,
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
        print_dry_run(selected_rows, prompt_text, args.output)
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

            try:
                api_input = build_api_input(prompt_text, str(row.get("input_text", "")))
                raw_response = generate_one(
                    client,
                    model=model,
                    api_input=api_input,
                    max_output_tokens=args.max_output_tokens,
                )
                error_message, parsed_response = validate_generation(
                    raw_response,
                    roles_array=list(row.get("roles_array", [])),
                )

                if error_message:
                    validation_failed_count += 1
                    record = build_output_record(
                        row,
                        prompt_path=args.prompt,
                        model=model,
                        generation_status="validation_failed",
                        raw_response=raw_response,
                        error_message=error_message,
                        parsed_response=parsed_response,
                    )
                    status = "validation_failed"
                else:
                    success_count += 1
                    record = build_output_record(
                        row,
                        prompt_path=args.prompt,
                        model=model,
                        generation_status="success",
                        raw_response=raw_response,
                        parsed_response=parsed_response,
                    )
                    status = "success"

            except Exception as exc:
                failed_count += 1
                record = build_output_record(
                    row,
                    prompt_path=args.prompt,
                    model=model,
                    generation_status="failed",
                    raw_response=raw_response,
                    error_message=f"{type(exc).__name__}: {exc}",
                )
                status = "failed"

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
