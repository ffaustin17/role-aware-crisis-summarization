"""Run a minimal OpenAI Responses API smoke test using local environment config."""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI


EXPECTED_RESPONSE = "OPENAI_SMOKE_TEST_OK"
DEFAULT_MODEL = "gpt-4o-mini"


def extract_response_text(response: object) -> str:
    """Return text from a Responses API result across SDK variants."""
    output_text = getattr(response, "output_text", None)

    if isinstance(output_text, str):
        return output_text.strip()

    return ""


def main() -> int:
    load_dotenv()

    api_key = os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)

    if not api_key:
        print(
            "OpenAI smoke test failed: missing OPENAI_KEY or OPENAI_API_KEY.",
            file=sys.stderr,
        )
        return 1

    try:
        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=model,
            input=f"Reply with exactly this text and nothing else: {EXPECTED_RESPONSE}",
            max_output_tokens=16,
        )
        response_text = extract_response_text(response)

        if response_text != EXPECTED_RESPONSE:
            print(
                "OpenAI smoke test failed: API call succeeded but returned "
                f"unexpected text: {response_text!r}",
                file=sys.stderr,
            )
            return 1

    except Exception as exc:
        print(
            f"OpenAI smoke test failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1

    print(f"OpenAI smoke test succeeded: {EXPECTED_RESPONSE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
