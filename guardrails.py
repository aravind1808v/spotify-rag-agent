"""
guardrails.py
Fast, heuristic-based guardrails — no LLM calls, no extra API cost.

Two layers:
  INPUT  guardrails – validate user inputs before the pipeline runs.
  OUTPUT guardrails – sanity-check generated outputs before returning them.

All public functions return a GuardrailResult. Call .raise_if_failed() to
stop the pipeline on hard errors, or check .warnings for soft signals.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

# ── Tuneable thresholds ─────────────────────────────────────────────────────────
MIN_QUERY_LENGTH = 3            # characters
MAX_QUERY_LENGTH = 500          # characters
MIN_JD_LENGTH = 50              # characters — raw text must be at least this
MIN_OUTPUT_LENGTH = 100         # characters — generated report must be at least this
LOW_RETRIEVAL_SCORE_THRESHOLD = 0.30  # below this similarity pct → low-quality warning
LOW_RETRIEVAL_WARN_FRACTION = 0.5     # if >50% of chunks are below threshold → warn

# Patterns that look like prompt-injection attempts in user-controlled inputs
_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.I),
    re.compile(r"you\s+are\s+now\s+a", re.I),
    re.compile(r"disregard\s+(the\s+)?above", re.I),
    re.compile(r"system\s*prompt", re.I),
    re.compile(r"<\s*system\s*>", re.I),
]

VALID_RESUME_EXTENSIONS = {".pdf", ".txt"}
VALID_JD_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}


# ── Result type ─────────────────────────────────────────────────────────────────

@dataclass
class GuardrailResult:
    passed: bool
    errors: list[str] = field(default_factory=list)    # hard failures → abort
    warnings: list[str] = field(default_factory=list)  # soft signals → log only

    def raise_if_failed(self) -> None:
        """Raise ValueError with all error messages if any hard errors exist."""
        if not self.passed:
            msg = "Guardrail violations:\n" + "\n".join(f"  • {e}" for e in self.errors)
            raise ValueError(msg)

    def print_warnings(self) -> None:
        """Print any soft warnings to stdout."""
        for w in self.warnings:
            print(f"  [WARN] {w}")


# ── Input guardrails ────────────────────────────────────────────────────────────

def validate_query(query: str) -> GuardrailResult:
    """
    Validate a Spotify search query.
    Checks: non-empty, length bounds, no prompt-injection patterns.
    """
    errors: list[str] = []
    warnings: list[str] = []

    q = query.strip() if query else ""

    if not q:
        errors.append("Query is empty. Please enter a topic or question.")
    elif len(q) < MIN_QUERY_LENGTH:
        errors.append(
            f"Query is too short ({len(q)} chars). "
            f"Please provide at least {MIN_QUERY_LENGTH} characters."
        )
    elif len(q) > MAX_QUERY_LENGTH:
        errors.append(
            f"Query is too long ({len(q)} chars, max {MAX_QUERY_LENGTH}). "
            "Please shorten your query."
        )

    for pattern in _INJECTION_PATTERNS:
        if pattern.search(q):
            errors.append(
                "Query contains a potential prompt-injection pattern and cannot be processed."
            )
            break

    if not errors and len(q.split()) == 1:
        warnings.append(
            "Single-word query may return broad results. "
            "Consider adding context (e.g. 'machine learning for beginners')."
        )

    return GuardrailResult(passed=len(errors) == 0, errors=errors, warnings=warnings)


def validate_resume_file(path: str) -> GuardrailResult:
    """
    Validate that the resume file exists and is a supported format.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if not path or not path.strip():
        errors.append("Resume path is empty.")
        return GuardrailResult(passed=False, errors=errors)

    if not os.path.isfile(path):
        errors.append(f"Resume file not found: '{path}'")
        return GuardrailResult(passed=False, errors=errors)

    ext = os.path.splitext(path)[1].lower()
    if ext not in VALID_RESUME_EXTENSIONS:
        errors.append(
            f"Unsupported resume format '{ext}'. "
            f"Supported formats: {', '.join(sorted(VALID_RESUME_EXTENSIONS))}"
        )

    file_size = os.path.getsize(path)
    if file_size == 0:
        errors.append(f"Resume file is empty: '{path}'")
    elif file_size > 10 * 1024 * 1024:  # 10 MB
        warnings.append(
            f"Resume file is large ({file_size // 1024} KB). "
            "Parsing may be slow for very large files."
        )

    return GuardrailResult(passed=len(errors) == 0, errors=errors, warnings=warnings)


def validate_jd_input(jd: str) -> GuardrailResult:
    """
    Validate a job description input (file path, URL, or raw text).
    For raw text: check minimum length.
    For file paths: check file existence.
    URLs are allowed through without checking (fetched at parse time).
    """
    errors: list[str] = []
    warnings: list[str] = []

    if not jd or not jd.strip():
        errors.append("Job description is empty. Provide a file path, URL, or raw JD text.")
        return GuardrailResult(passed=False, errors=errors)

    jd = jd.strip()

    # URL → defer to parse time
    if jd.startswith("http://") or jd.startswith("https://"):
        return GuardrailResult(passed=True, warnings=["JD is a URL — will be fetched at runtime."])

    # File path — check by existence first, then by extension
    ext = os.path.splitext(jd)[1].lower()
    looks_like_file = ext in VALID_JD_EXTENSIONS

    if os.path.isfile(jd):
        size = os.path.getsize(jd)
        if size == 0:
            errors.append(f"JD file is empty: '{jd}'")
        return GuardrailResult(passed=len(errors) == 0, errors=errors, warnings=warnings)

    if looks_like_file:
        # Has a file extension but isfile() returned False — path not found
        errors.append(
            f"JD file not found: '{jd}'. "
            "Check the path is correct and you are running from the right directory."
        )
        return GuardrailResult(passed=False, errors=errors)

    # Raw text
    if len(jd) < MIN_JD_LENGTH:
        errors.append(
            f"Job description text is too short ({len(jd)} chars, min {MIN_JD_LENGTH}). "
            "Provide more detail or use a file path."
        )

    for pattern in _INJECTION_PATTERNS:
        if pattern.search(jd):
            errors.append(
                "Job description contains a potential prompt-injection pattern."
            )
            break

    return GuardrailResult(passed=len(errors) == 0, errors=errors, warnings=warnings)


# ── Retrieval quality guardrail ─────────────────────────────────────────────────

def check_retrieval_quality(
    scores: list[float],
    scores_are_cohere: bool = False,
    threshold: float = LOW_RETRIEVAL_SCORE_THRESHOLD,
    warn_fraction: float = LOW_RETRIEVAL_WARN_FRACTION,
) -> GuardrailResult:
    """
    Warn if the retrieved chunk set is low quality.

    For FAISS similarity scores (0–100 pct): uses an absolute threshold —
    warns if more than `warn_fraction` of chunks fall below `threshold * 100`.

    For Cohere rerank scores (0–1): uses a distribution-aware check because
    Cohere scores are relative and an absolute threshold is meaningless.
    Two conditions trigger a warning:
      - Top score < 0.05: nothing in the corpus matched the query at all.
      - More than half the chunks score below 30% of the top score:
        the reranker found a few decent results but most are irrelevant.

    Args:
        scores:             Similarity percentages (0–100) for FAISS, or raw
                            Cohere rerank scores (0–1) when scores_are_cohere=True.
        scores_are_cohere:  Set True when scores come from Cohere rerank.
        threshold:          FAISS mode only — low-quality floor (0–1, default 0.30).
        warn_fraction:      FAISS mode only — fraction below threshold to trigger warn.
    """
    warnings: list[str] = []

    if not scores:
        return GuardrailResult(
            passed=True,
            warnings=["No retrieval scores available to evaluate quality."]
        )

    if scores_are_cohere:
        top_score = max(scores)
        avg_score = sum(scores) / len(scores)

        if top_score < 0.05:
            warnings.append(
                f"Very low Cohere rerank scores (top: {top_score:.3f}, avg: {avg_score:.3f}). "
                "The retrieved content may not be relevant to your query."
            )
        else:
            relative_floor = top_score * 0.30
            low_quality = [s for s in scores if s < relative_floor]
            if len(low_quality) > len(scores) / 2:
                warnings.append(
                    f"Uneven retrieval quality: {len(low_quality)}/{len(scores)} chunks "
                    f"scored below 30% of the top score ({top_score:.3f}). "
                    f"Average: {avg_score:.3f}. Some retrieved chunks may be off-topic."
                )
    else:
        threshold_pct = threshold * 100
        low_quality = [s for s in scores if s < threshold_pct]
        fraction_low = len(low_quality) / len(scores)
        avg_score = sum(scores) / len(scores)

        if fraction_low > warn_fraction:
            warnings.append(
                f"Low retrieval quality: {len(low_quality)}/{len(scores)} chunks "
                f"scored below {threshold_pct:.0f}% similarity. "
                f"Average score: {avg_score:.1f}%. "
                "Results may not be highly relevant to your query."
            )

    return GuardrailResult(passed=True, warnings=warnings)


# ── Output guardrails ───────────────────────────────────────────────────────────

def check_output_completeness(
    output: str,
    min_length: int = MIN_OUTPUT_LENGTH,
    expected_sections: list[str] | None = None,
) -> GuardrailResult:
    """
    Ensure the generated output is substantive and contains expected structure.

    Args:
        output:            The LLM-generated report string.
        min_length:        Minimum character length for a valid output.
        expected_sections: Optional list of section headings to check for presence.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if not output or not output.strip():
        errors.append("Generated output is empty.")
        return GuardrailResult(passed=False, errors=errors)

    if len(output.strip()) < min_length:
        errors.append(
            f"Generated output is suspiciously short ({len(output.strip())} chars, "
            f"expected at least {min_length}). The pipeline may have failed silently."
        )

    if expected_sections:
        for section in expected_sections:
            if section.lower() not in output.lower():
                warnings.append(
                    f"Expected section '{section}' not found in output. "
                    "The report may be incomplete."
                )

    # Detect common LLM refusal / error phrases
    refusal_signals = [
        "i cannot", "i'm unable", "i don't have access",
        "as an ai", "i apologize, but i cannot",
    ]
    output_lower = output.lower()
    for signal in refusal_signals:
        if signal in output_lower:
            warnings.append(
                f"Output contains a possible LLM refusal phrase ('{signal}'). "
                "Review the report for completeness."
            )
            break

    return GuardrailResult(passed=len(errors) == 0, errors=errors, warnings=warnings)
