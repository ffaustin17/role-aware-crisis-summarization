"""Score role-aware summary predictions with transparent reward components."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_INPUT_PATH = Path("data/modeling/t5_baseline_v1/test_predictions.jsonl")
DEFAULT_OUTPUT_JSONL_PATH = Path("data/rewards/t5_baseline_v1_reward_scores.jsonl")
DEFAULT_SUMMARY_CSV_PATH = Path("reports/tables/t5_baseline_reward_summary.csv")
DEFAULT_SUMMARY_MD_PATH = Path("reports/tables/t5_baseline_reward_summary.md")
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_RELEVANCE_BACKEND = "sentence-transformer"

COMPONENT_WEIGHTS = {
    "relevance": 0.35,
    "factuality": 0.25,
    "role_coverage": 0.20,
    "urgency": 0.20,
}

ROLE_CRITERIA = {
    "EMS": {
        "injury_casualty": [
            "injured",
            "injuries",
            "injury",
            "wounded",
            "casualties",
            "fatalities",
            "dead",
            "death toll",
            "killed",
        ],
        "medical_response": [
            "medical",
            "medical support",
            "medical emergency",
            "medical emergencies",
            "patient",
            "ambulance",
            "hospital",
            "mmu",
            "cert",
        ],
        "triage_transport": [
            "triage",
            "transport",
            "patient transport",
            "prioritize medical",
        ],
        "urgency_vulnerability": [
            "urgent",
            "urgency",
            "critical",
            "vulnerable",
            "rescue",
            "trapped",
        ],
    },
    "Firefighter": {
        "fire_spread": [
            "fire",
            "fires",
            "wildfire",
            "flames",
            "smoke",
            "haze",
            "spread",
            "fire spread",
            "growing",
        ],
        "containment": [
            "containment",
            "contain",
            "contained",
            "uncontained",
            "containment strategies",
        ],
        "hazardous_materials": [
            "hazardous",
            "hazardous material",
            "hazmat",
            "material exposure",
            "chemical",
            "gas",
        ],
        "structural_rescue": [
            "structural",
            "collapse",
            "trapped",
            "search rescue",
            "rescue operations",
            "usar",
        ],
        "smoke_air_quality": [
            "air quality",
            "psi",
            "smoke exposure",
            "haze",
        ],
    },
    "Police": {
        "threat_security": [
            "threat",
            "threats",
            "safety threat",
            "public safety",
            "security",
        ],
        "crowd_control": [
            "crowd",
            "crowd control",
        ],
        "evacuation_access": [
            "evacuation",
            "evacuate",
            "evacuations",
            "access",
            "traffic",
            "road closure",
            "blocked roads",
            "teu",
        ],
        "scene_security": [
            "scene security",
            "ensure scene",
            "dispatch",
            "dcc",
        ],
        "criminal_activity": [
            "criminal activity",
            "arrest",
            "arrested",
            "investigate",
            "probe",
            "unrest",
            "protest",
        ],
    },
}

URGENCY_TERMS = [
    "urgent",
    "urgency",
    "critical",
    "emergency",
    "injured",
    "injuries",
    "casualties",
    "fatalities",
    "dead",
    "killed",
    "trapped",
    "evacuation",
    "evacuate",
    "evacuations",
    "threat",
    "threats",
    "hazardous",
    "hazmat",
    "fire spread",
    "uncontained",
    "collapse",
    "explosion",
    "shooting",
    "crash",
]

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "while",
    "with",
}

TOKEN_PATTERN = re.compile(r"[a-z0-9][a-z0-9'-]*", re.IGNORECASE)
NUMBER_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\b")
IDENTIFIER_PATTERN = re.compile(r"\b[A-Z]{2,}\b")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for reward scoring."""
    parser = argparse.ArgumentParser(
        description="Score role-aware summary predictions with reward components."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_JSONL_PATH)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV_PATH)
    parser.add_argument("--summary-md", type=Path, default=DEFAULT_SUMMARY_MD_PATH)
    parser.add_argument(
        "--relevance-backend",
        choices=["sentence-transformer", "lexical"],
        default=DEFAULT_RELEVANCE_BACKEND,
    )
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--max-records", type=int, default=None)
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
    """Return a stripped string for scoring."""
    if value is None:
        return ""
    return str(value).strip()


def split_roles(role_label: str) -> list[str]:
    """Parse exact role labels like Police/EMS into known responder roles."""
    roles = []
    for role in role_label.split("/"):
        role = role.strip()
        if role in ROLE_CRITERIA:
            roles.append(role)
    return roles


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercased content tokens."""
    tokens = [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]
    return [token for token in tokens if len(token) > 2 and token not in STOPWORDS]


def contains_phrase(text: str, phrase: str) -> bool:
    """Return true if phrase appears as a loose lowercase substring."""
    normalized_text = f" {text.lower()} "
    normalized_phrase = f" {phrase.lower()} "
    return normalized_phrase in normalized_text


def matched_terms(text: str, terms: list[str]) -> list[str]:
    """Return configured terms found in text."""
    return [term for term in terms if contains_phrase(text, term)]


def build_source_context(record: dict[str, Any]) -> str:
    """Build source context used for grounding and relevance."""
    parts = [
        normalize_text(record.get("tweet_text")),
        normalize_text(record.get("input_text")),
        normalize_text(record.get("disaster_type")),
        normalize_text(record.get("role")),
        normalize_text(record.get("information_type")),
    ]
    return "\n".join(part for part in parts if part)


def lexical_similarity(source_text: str, candidate_text: str) -> float:
    """Compute a simple token-overlap relevance fallback."""
    source_tokens = set(tokenize(source_text))
    candidate_tokens = set(tokenize(candidate_text))
    if not source_tokens or not candidate_tokens:
        return 0.0
    intersection = source_tokens & candidate_tokens
    return len(intersection) / math.sqrt(len(source_tokens) * len(candidate_tokens))


class RelevanceScorer:
    """Compute relevance with SentenceTransformer or lexical fallback."""

    def __init__(self, backend: str, embedding_model: str) -> None:
        self.backend = backend
        self.model = None
        if backend == "sentence-transformer":
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(embedding_model)

    def score_batch(
        self,
        source_texts: list[str],
        candidate_texts: list[str],
    ) -> list[float]:
        """Score source/candidate pairs on a 0.0 to 1.0 scale."""
        if self.backend == "lexical":
            return [
                lexical_similarity(source_text, candidate_text)
                for source_text, candidate_text in zip(
                    source_texts,
                    candidate_texts,
                    strict=True,
                )
            ]

        assert self.model is not None
        source_embeddings = self.model.encode(source_texts, normalize_embeddings=True)
        candidate_embeddings = self.model.encode(
            candidate_texts,
            normalize_embeddings=True,
        )
        scores: list[float] = []
        for source_embedding, candidate_embedding in zip(
            source_embeddings,
            candidate_embeddings,
            strict=True,
        ):
            cosine = float((source_embedding * candidate_embedding).sum())
            scores.append(max(0.0, min(1.0, (cosine + 1.0) / 2.0)))
        return scores


def score_factuality_proxy(source_text: str, candidate_text: str) -> tuple[float, dict[str, Any]]:
    """Score whether candidate tokens and identifiers are grounded in the source."""
    source_tokens = set(tokenize(source_text))
    candidate_tokens = tokenize(candidate_text)
    candidate_content = [
        token for token in candidate_tokens if token not in {"assess", "prepare", "monitor"}
    ]

    if not candidate_content:
        token_score = 0.0
        unsupported_tokens: list[str] = []
    else:
        supported = [token for token in candidate_content if token in source_tokens]
        unsupported_tokens = sorted(set(candidate_content) - source_tokens)
        token_score = len(supported) / len(candidate_content)

    source_numbers = set(NUMBER_PATTERN.findall(source_text))
    candidate_numbers = set(NUMBER_PATTERN.findall(candidate_text))
    unsupported_numbers = sorted(candidate_numbers - source_numbers)
    number_score = 1.0 if not unsupported_numbers else 0.0

    source_ids = set(IDENTIFIER_PATTERN.findall(source_text))
    candidate_ids = set(IDENTIFIER_PATTERN.findall(candidate_text))
    unsupported_ids = sorted(candidate_ids - source_ids)
    identifier_score = 1.0 if not unsupported_ids else 0.0

    score = (0.70 * token_score) + (0.15 * number_score) + (0.15 * identifier_score)
    details = {
        "unsupported_terms": unsupported_tokens[:20],
        "unsupported_numbers": unsupported_numbers,
        "unsupported_identifiers": unsupported_ids,
    }
    return max(0.0, min(1.0, score)), details


def score_role_coverage(
    source_text: str,
    candidate_text: str,
    roles: list[str],
) -> tuple[float, dict[str, Any]]:
    """Score candidate coverage of role criteria with source evidence."""
    applicable_categories = 0
    covered_categories = 0
    details: dict[str, Any] = {}

    for role in roles:
        role_details: dict[str, Any] = {}
        for category, terms in ROLE_CRITERIA[role].items():
            source_matches = matched_terms(source_text, terms)
            if not source_matches:
                continue

            applicable_categories += 1
            candidate_matches = matched_terms(candidate_text, terms)
            if candidate_matches:
                covered_categories += 1

            role_details[category] = {
                "source_terms": source_matches,
                "candidate_terms": candidate_matches,
                "covered": bool(candidate_matches),
            }

        details[role] = role_details

    if applicable_categories == 0:
        return 0.5, details

    return covered_categories / applicable_categories, details


def score_urgency(source_text: str, candidate_text: str) -> tuple[float, dict[str, Any]]:
    """Score whether urgent source signals are reflected in the candidate."""
    source_matches = matched_terms(source_text, URGENCY_TERMS)
    candidate_matches = matched_terms(candidate_text, URGENCY_TERMS)

    if not source_matches:
        score = 0.5 if not candidate_matches else 0.4
    else:
        score = len(set(source_matches) & set(candidate_matches)) / len(set(source_matches))

    details = {
        "source_terms": source_matches,
        "candidate_terms": candidate_matches,
    }
    return max(0.0, min(1.0, score)), details


def composite_reward(component_scores: dict[str, float]) -> float:
    """Combine component scores with the project reward weights."""
    return sum(
        component_scores[component] * weight
        for component, weight in COMPONENT_WEIGHTS.items()
    )


def score_records(
    records: list[dict[str, Any]],
    relevance_scorer: RelevanceScorer,
) -> list[dict[str, Any]]:
    """Score all records and return traceable reward records."""
    source_texts = [build_source_context(record) for record in records]
    candidate_texts = [
        normalize_text(record.get("prediction_text") or record.get("final_base_summary_text"))
        for record in records
    ]
    relevance_scores = relevance_scorer.score_batch(source_texts, candidate_texts)

    scored_records: list[dict[str, Any]] = []
    for record, source_text, candidate_text, relevance_score in zip(
        records,
        source_texts,
        candidate_texts,
        relevance_scores,
        strict=True,
    ):
        roles = split_roles(normalize_text(record.get("role")))
        factuality_score, factuality_details = score_factuality_proxy(
            source_text,
            candidate_text,
        )
        role_coverage_score, role_coverage_details = score_role_coverage(
            source_text,
            candidate_text,
            roles,
        )
        urgency_score, urgency_details = score_urgency(source_text, candidate_text)
        component_scores = {
            "relevance": relevance_score,
            "factuality": factuality_score,
            "role_coverage": role_coverage_score,
            "urgency": urgency_score,
        }
        reward = composite_reward(component_scores)

        scored_records.append(
            {
                "tweet_id": record.get("tweet_id"),
                "source_row_id": record.get("source_row_id"),
                "role": record.get("role"),
                "disaster_type": record.get("disaster_type"),
                "information_type": record.get("information_type"),
                "prediction_text": candidate_text,
                "target_text": record.get("target_text"),
                "reward": reward,
                "component_scores": component_scores,
                "component_weights": COMPONENT_WEIGHTS,
                "roles_scored": roles,
                "role_coverage_details": role_coverage_details,
                "urgency_details": urgency_details,
                "factuality_details": factuality_details,
            }
        )

    return scored_records


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    """Write scored records to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def summarize_scores(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create overall and per-role reward summaries."""
    rows: list[dict[str, Any]] = []

    def add_summary(group_name: str, group_records: list[dict[str, Any]]) -> None:
        if not group_records:
            return
        rows.append(
            {
                "group": group_name,
                "num_examples": len(group_records),
                "reward_mean": mean(record["reward"] for record in group_records),
                "relevance_mean": mean(
                    record["component_scores"]["relevance"] for record in group_records
                ),
                "factuality_mean": mean(
                    record["component_scores"]["factuality"] for record in group_records
                ),
                "role_coverage_mean": mean(
                    record["component_scores"]["role_coverage"]
                    for record in group_records
                ),
                "urgency_mean": mean(
                    record["component_scores"]["urgency"] for record in group_records
                ),
            }
        )

    add_summary("overall", records)
    role_to_records: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        role_to_records.setdefault(str(record["role"]), []).append(record)
    for role, role_records in sorted(role_to_records.items()):
        add_summary(role, role_records)

    return rows


def write_summary_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Write reward summary rows as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "group",
        "num_examples",
        "reward_mean",
        "relevance_mean",
        "factuality_mean",
        "role_coverage_mean",
        "urgency_mean",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    """Write reward summary rows as a Markdown table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        file.write(
            "| group | num_examples | reward_mean | relevance_mean | factuality_mean | role_coverage_mean | urgency_mean |\n"
        )
        file.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            file.write(
                f"| {row['group']} | {row['num_examples']} | "
                f"{row['reward_mean']:.6f} | {row['relevance_mean']:.6f} | "
                f"{row['factuality_mean']:.6f} | {row['role_coverage_mean']:.6f} | "
                f"{row['urgency_mean']:.6f} |\n"
            )


def main() -> int:
    """Score predictions and write reward artifacts."""
    args = parse_args()
    records = read_jsonl(args.input)
    if args.max_records is not None:
        records = records[: args.max_records]

    relevance_scorer = RelevanceScorer(
        backend=args.relevance_backend,
        embedding_model=args.embedding_model,
    )
    scored_records = score_records(records, relevance_scorer)
    summary_rows = summarize_scores(scored_records)

    write_jsonl(scored_records, args.output)
    write_summary_csv(summary_rows, args.summary_csv)
    write_summary_markdown(summary_rows, args.summary_md)

    print(f"Scored records: {len(scored_records)}")
    print(f"Wrote reward scores to: {args.output}")
    print(f"Wrote reward summary CSV to: {args.summary_csv}")
    print(f"Wrote reward summary Markdown to: {args.summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
