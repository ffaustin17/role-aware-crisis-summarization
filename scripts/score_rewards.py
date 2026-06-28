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
DEFAULT_FACTUALITY_BACKEND = "proxy"
DEFAULT_MINICHECK_MODEL = "flan-t5-large"
DEFAULT_MINICHECK_CACHE_DIR = Path("models/minicheck")
TWEET_RELEVANCE_WEIGHT = 0.70
CONTEXT_RELEVANCE_WEIGHT = 0.30

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
            "medical aid",
            "medical assistance",
            "medical support",
            "emergency medical",
            "medical emergency",
            "medical emergencies",
            "patient",
            "patient care",
            "treatment",
            "care",
            "ambulance",
            "ambulance response",
            "hospital",
            "mmu",
            "cert",
        ],
        "triage_transport": [
            "triage",
            "transport",
            "patient transport",
            "prioritize medical",
            "mass casualty",
        ],
        "urgency_vulnerability": [
            "urgent",
            "urgency",
            "critical",
            "vulnerable",
            "rescue",
            "trapped",
            "hurt",
            "people hurt",
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
            "fire control",
            "fc",
            "firefighting",
            "fire crews",
            "fire crew",
        ],
        "hazardous_materials": [
            "hazardous",
            "hazardous material",
            "hazmat",
            "material exposure",
            "chemical",
            "gas",
            "toxic",
            "fumes",
            "smoke inhalation",
        ],
        "structural_rescue": [
            "structural",
            "structural safety",
            "collapse",
            "trapped",
            "search and rescue",
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
            "law enforcement",
            "public order",
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
            "access control",
            "traffic",
            "traffic control",
            "road closure",
            "road closures",
            "blocked roads",
            "teu",
        ],
        "scene_security": [
            "scene security",
            "ensure scene",
            "secure the area",
            "perimeter",
            "dispatch",
            "dcc",
        ],
        "criminal_activity": [
            "criminal activity",
            "arrest",
            "arrested",
            "investigate",
            "criminal investigation",
            "crime prevention",
            "probe",
            "unrest",
            "protest",
            "cpu",
        ],
    },
}

URGENCY_CRITERIA = {
    "casualty_injury": [
        "injured",
        "injuries",
        "injury",
        "wounded",
        "hurt",
        "casualties",
        "fatalities",
        "dead",
        "death",
        "death toll",
        "killed",
        "feared dead",
    ],
    "rescue_evacuation": [
        "trapped",
        "stranded",
        "rescue",
        "evacuation",
        "evacuate",
        "evacuated",
        "evacuations",
        "shelter",
    ],
    "active_hazard": [
        "hazardous",
        "hazmat",
        "toxic",
        "chemical",
        "gas",
        "fire spread",
        "uncontained",
        "fire",
        "wildfire",
        "flames",
        "smoke",
        "collapse",
        "explosion",
        "shooting",
        "crash",
        "flood",
        "flooding",
        "rising water",
    ],
    "severity_threat": [
        "urgent",
        "urgency",
        "critical",
        "emergency",
        "severe",
        "immediate",
        "threat",
        "threats",
        "danger",
        "dangerous",
    ],
}

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
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")
PHRASE_NORMALIZE_PATTERN = re.compile(r"[^a-z0-9]+")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for reward scoring."""
    parser = argparse.ArgumentParser(
        description="Score role-aware summary predictions with reward components."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--summary-md", type=Path, default=None)
    parser.add_argument(
        "--relevance-backend",
        choices=["sentence-transformer", "lexical"],
        default=DEFAULT_RELEVANCE_BACKEND,
    )
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument(
        "--factuality-backend",
        choices=["proxy", "minicheck"],
        default=DEFAULT_FACTUALITY_BACKEND,
        help="Use the local grounding proxy or MiniCheck for factuality.",
    )
    parser.add_argument(
        "--minicheck-model",
        default=DEFAULT_MINICHECK_MODEL,
        help="MiniCheck model name, used only with --factuality-backend minicheck.",
    )
    parser.add_argument(
        "--minicheck-cache-dir",
        type=Path,
        default=DEFAULT_MINICHECK_CACHE_DIR,
        help="Checkpoint/cache directory for MiniCheck models.",
    )
    parser.add_argument("--max-records", type=int, default=None)
    args = parser.parse_args()
    if args.output is None:
        args.output = default_output_path(args.factuality_backend)
    if args.summary_csv is None:
        args.summary_csv = default_summary_csv_path(args.factuality_backend)
    if args.summary_md is None:
        args.summary_md = default_summary_md_path(args.factuality_backend)
    return args


def add_stem_suffix(path: Path, suffix: str) -> Path:
    """Return path with a suffix added before the file extension."""
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def default_output_path(factuality_backend: str) -> Path:
    """Choose a backend-aware default reward JSONL path."""
    if factuality_backend == "minicheck":
        return add_stem_suffix(DEFAULT_OUTPUT_JSONL_PATH, "_minicheck")
    return DEFAULT_OUTPUT_JSONL_PATH


def default_summary_csv_path(factuality_backend: str) -> Path:
    """Choose a backend-aware default reward summary CSV path."""
    if factuality_backend == "minicheck":
        return add_stem_suffix(DEFAULT_SUMMARY_CSV_PATH, "_minicheck")
    return DEFAULT_SUMMARY_CSV_PATH


def default_summary_md_path(factuality_backend: str) -> Path:
    """Choose a backend-aware default reward summary Markdown path."""
    if factuality_backend == "minicheck":
        return add_stem_suffix(DEFAULT_SUMMARY_MD_PATH, "_minicheck")
    return DEFAULT_SUMMARY_MD_PATH


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


def split_sentences(text: str) -> list[str]:
    """Split a generated summary into sentence-level claims for MiniCheck."""
    normalized = normalize_text(text)
    if not normalized:
        return []
    sentences = [
        sentence.strip()
        for sentence in SENTENCE_SPLIT_PATTERN.split(normalized)
        if sentence.strip()
    ]
    return sentences or [normalized]


def normalize_for_phrase_matching(value: str) -> str:
    """Normalize text so phrase matching is stable across punctuation."""
    lowered = value.lower()
    alphanumeric = PHRASE_NORMALIZE_PATTERN.sub(" ", lowered)
    return " ".join(alphanumeric.split())


def contains_phrase(text: str, phrase: str) -> bool:
    """Return true if a normalized phrase appears with token boundaries."""
    normalized_text = normalize_for_phrase_matching(text)
    normalized_phrase = normalize_for_phrase_matching(phrase)
    if not normalized_text or not normalized_phrase:
        return False

    pattern = rf"(?<![a-z0-9]){re.escape(normalized_phrase)}(?![a-z0-9])"
    return re.search(pattern, normalized_text) is not None


def matched_terms(text: str, terms: list[str]) -> list[str]:
    """Return configured terms found in text."""
    return [term for term in terms if contains_phrase(text, term)]


def matched_terms_by_category(
    text: str,
    criteria: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Return matched terms grouped by configured category."""
    matches_by_category: dict[str, list[str]] = {}
    for category, terms in criteria.items():
        category_matches = matched_terms(text, terms)
        if category_matches:
            matches_by_category[category] = category_matches
    return matches_by_category


def extract_tweet_from_input_text(input_text: str) -> str:
    """Extract the Tweet line from a generated input_text field."""
    for line in input_text.splitlines():
        if line.strip().lower().startswith("tweet:"):
            return line.split(":", maxsplit=1)[1].strip()
    return ""


def remove_tweet_from_input_text(input_text: str) -> str:
    """Return input_text metadata without the source Tweet line."""
    lines = [
        line.strip()
        for line in input_text.splitlines()
        if line.strip() and not line.strip().lower().startswith("tweet:")
    ]
    return "\n".join(lines)


def unique_nonempty(parts: list[str]) -> list[str]:
    """Return non-empty strings while preserving first-seen order."""
    seen = set()
    values = []
    for part in parts:
        normalized = normalize_text(part)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        values.append(normalized)
    return values


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


def build_relevance_sources(record: dict[str, Any]) -> dict[str, Any]:
    """Build tweet-dominant relevance sources from a prediction record."""
    input_text = normalize_text(record.get("input_text"))
    tweet_text = normalize_text(record.get("tweet_text")) or extract_tweet_from_input_text(
        input_text
    )
    context_parts = unique_nonempty(
        [
            normalize_text(record.get("disaster_type")),
            normalize_text(record.get("role")),
            normalize_text(record.get("information_type")),
            remove_tweet_from_input_text(input_text),
        ]
    )
    context_text = "\n".join(context_parts)
    combined_text = build_source_context(record)
    return {
        "tweet_text": tweet_text,
        "context_text": context_text,
        "combined_text": combined_text,
        "tweet_source_available": bool(tweet_text),
        "context_source_available": bool(context_text),
    }


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

    def score_weighted_batch(
        self,
        relevance_sources: list[dict[str, Any]],
        candidate_texts: list[str],
    ) -> tuple[list[float], list[dict[str, Any]]]:
        """Score relevance with source tweet weighted above metadata context."""
        tweet_source_texts = []
        context_source_texts = []
        for sources in relevance_sources:
            if sources["tweet_source_available"]:
                tweet_source_texts.append(sources["tweet_text"])
            else:
                tweet_source_texts.append(sources["combined_text"])

            if sources["context_source_available"]:
                context_source_texts.append(sources["context_text"])
            else:
                context_source_texts.append(sources["tweet_text"])

        tweet_scores = self.score_batch(tweet_source_texts, candidate_texts)
        context_scores = self.score_batch(context_source_texts, candidate_texts)

        relevance_scores = []
        relevance_details = []
        for sources, tweet_score, context_score in zip(
            relevance_sources,
            tweet_scores,
            context_scores,
            strict=True,
        ):
            tweet_available = sources["tweet_source_available"]
            context_available = sources["context_source_available"]
            fallback_source = None
            if tweet_available and context_available:
                tweet_weight = TWEET_RELEVANCE_WEIGHT
                context_weight = CONTEXT_RELEVANCE_WEIGHT
                relevance_score = (
                    tweet_weight * tweet_score
                    + context_weight * context_score
                )
            elif tweet_available:
                tweet_weight = 1.0
                context_weight = 0.0
                relevance_score = tweet_score
                context_score = None
            else:
                tweet_weight = 0.0
                context_weight = 0.0
                relevance_score = tweet_score
                tweet_score = None
                fallback_source = "combined_source_context"
                if not context_available:
                    context_score = None

            relevance_scores.append(relevance_score)
            relevance_details.append(
                {
                    "tweet_relevance": tweet_score,
                    "context_relevance": context_score,
                    "tweet_weight": tweet_weight,
                    "context_weight": context_weight,
                    "tweet_source_available": tweet_available,
                    "context_source_available": context_available,
                    "fallback_source": fallback_source,
                }
            )

        return relevance_scores, relevance_details


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


def as_float(value: Any) -> float:
    """Convert MiniCheck probability outputs to plain floats."""
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def as_list(value: Any) -> list[Any]:
    """Convert tensor-like MiniCheck outputs to plain Python lists."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        converted = value.tolist()
        return converted if isinstance(converted, list) else [converted]
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def normalize_probability(value: Any) -> float:
    """Extract a support probability from common MiniCheck return shapes."""
    values = as_list(value)
    if not values:
        return 0.0
    if len(values) >= 2:
        return as_float(values[1])
    return as_float(values[0])


def normalize_label(value: Any) -> Any:
    """Convert MiniCheck labels to JSON-serializable values."""
    values = as_list(value)
    if len(values) == 1:
        single_value = values[0]
        if isinstance(single_value, float) and single_value.is_integer():
            return int(single_value)
        return single_value
    return values


def positive_label_probability(_pred_label: Any, raw_prob: Any) -> float:
    """Return MiniCheck's support probability when labels and probs are paired."""
    probabilities = as_list(raw_prob)
    if len(probabilities) >= 2:
        return normalize_probability(probabilities)

    return as_float(probabilities[0]) if probabilities else 0.0


def sentence_score_records(
    claims: list[str],
    pred_labels: Any,
    raw_probs: Any,
) -> tuple[list[float], list[dict[str, Any]]]:
    """Build MiniCheck sentence-level score details."""
    labels = as_list(pred_labels)
    probabilities = as_list(raw_probs)
    support_scores = []
    sentence_scores = []
    for index, claim in enumerate(claims):
        pred_label = labels[index] if index < len(labels) else None
        raw_prob = probabilities[index] if index < len(probabilities) else 0.0
        support_probability = max(
            0.0,
            min(1.0, positive_label_probability(pred_label, raw_prob)),
        )
        support_scores.append(support_probability)
        sentence_scores.append(
            {
                "claim": claim,
                "predicted_label": normalize_label(pred_label),
                "support_probability": support_probability,
            }
        )
    return support_scores, sentence_scores


class FactualityScorer:
    """Score factual grounding with the proxy scorer or optional MiniCheck."""

    def __init__(
        self,
        backend: str,
        minicheck_model: str,
        minicheck_cache_dir: Path,
    ) -> None:
        self.backend = backend
        self.model_name = minicheck_model
        self.model = None
        if backend == "minicheck":
            try:
                from minicheck.minicheck import MiniCheck
            except ImportError as exc:
                raise ImportError(
                    "MiniCheck is not installed. Install it in the scoring "
                    "environment or use --factuality-backend proxy."
                ) from exc

            self.model = MiniCheck(
                model_name=minicheck_model,
                cache_dir=str(minicheck_cache_dir),
            )

    def score(self, source_text: str, candidate_text: str) -> tuple[float, dict[str, Any]]:
        """Return a factuality score and traceable backend details."""
        if self.backend == "proxy":
            score, details = score_factuality_proxy(source_text, candidate_text)
            details["backend"] = "proxy"
            return score, details

        return self.score_minicheck(source_text, candidate_text)

    def score_minicheck(
        self,
        source_text: str,
        candidate_text: str,
    ) -> tuple[float, dict[str, Any]]:
        """Score sentence-level summary claims against the source context."""
        assert self.model is not None
        claims = split_sentences(candidate_text)
        if not claims:
            return 0.0, {
                "backend": "minicheck",
                "model": self.model_name,
                "sentence_scores": [],
            }

        docs = [source_text] * len(claims)
        pred_labels, raw_probs, _, _ = self.model.score(docs=docs, claims=claims)
        support_scores, sentence_scores = sentence_score_records(
            claims,
            pred_labels,
            raw_probs,
        )

        return mean(support_scores), {
            "backend": "minicheck",
            "model": self.model_name,
            "sentence_scores": sentence_scores,
        }


def score_role_coverage(
    source_text: str,
    candidate_text: str,
    roles: list[str],
) -> tuple[float, dict[str, Any]]:
    """Score candidate coverage of role criteria with source evidence."""
    applicable_categories = 0
    covered_categories = 0
    details: dict[str, Any] = {
        "applicable_category_count": 0,
        "covered_category_count": 0,
        "score_reason": "",
        "roles": {},
    }

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

        details["roles"][role] = role_details

    if applicable_categories == 0:
        details["score_reason"] = "neutral_no_applicable_source_role_evidence"
        return 0.5, details

    score = covered_categories / applicable_categories
    details["applicable_category_count"] = applicable_categories
    details["covered_category_count"] = covered_categories
    details["score_reason"] = "covered_applicable_source_role_categories"
    return score, details


def score_urgency(source_text: str, candidate_text: str) -> tuple[float, dict[str, Any]]:
    """Score whether source urgency concepts are reflected in the candidate."""
    source_matches_by_category = matched_terms_by_category(source_text, URGENCY_CRITERIA)
    candidate_matches_by_category = matched_terms_by_category(
        candidate_text,
        URGENCY_CRITERIA,
    )
    source_categories = list(source_matches_by_category.keys())
    candidate_categories = list(candidate_matches_by_category.keys())
    covered_categories = [
        category for category in source_categories if category in candidate_matches_by_category
    ]

    if not source_categories:
        if not candidate_categories:
            score = 0.5
            score_reason = "neutral_no_source_or_candidate_urgency_evidence"
        else:
            score = 0.4
            score_reason = "candidate_adds_urgency_without_source_evidence"
    else:
        score = len(covered_categories) / len(source_categories)
        score_reason = "covered_source_urgency_categories"

    details = {
        "source_categories": source_categories,
        "candidate_categories": candidate_categories,
        "covered_categories": covered_categories,
        "source_terms_by_category": source_matches_by_category,
        "candidate_terms_by_category": candidate_matches_by_category,
        "score_reason": score_reason,
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
    factuality_scorer: FactualityScorer,
) -> list[dict[str, Any]]:
    """Score all records and return traceable reward records."""
    source_texts = [build_source_context(record) for record in records]
    relevance_sources = [build_relevance_sources(record) for record in records]
    candidate_texts = [
        normalize_text(record.get("prediction_text") or record.get("final_base_summary_text"))
        for record in records
    ]
    relevance_scores, relevance_details_list = relevance_scorer.score_weighted_batch(
        relevance_sources,
        candidate_texts,
    )

    scored_records: list[dict[str, Any]] = []
    for (
        record,
        source_text,
        candidate_text,
        relevance_score,
        relevance_details,
    ) in zip(
        records,
        source_texts,
        candidate_texts,
        relevance_scores,
        relevance_details_list,
        strict=True,
    ):
        roles = split_roles(normalize_text(record.get("role")))
        factuality_score, factuality_details = factuality_scorer.score(
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
                "relevance_details": relevance_details,
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
    factuality_scorer = FactualityScorer(
        backend=args.factuality_backend,
        minicheck_model=args.minicheck_model,
        minicheck_cache_dir=args.minicheck_cache_dir,
    )
    scored_records = score_records(records, relevance_scorer, factuality_scorer)
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
