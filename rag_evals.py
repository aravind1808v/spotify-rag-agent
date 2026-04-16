"""
rag_evals.py
LLM-as-judge evaluation metrics for the RAG pipelines.

Five metrics are implemented:

  1. context_relevance    – Are the retrieved chunks relevant to the query?
  2. faithfulness         – Is the answer grounded in the retrieved context (no hallucinations)?
  3. answer_relevance     – Does the answer actually address the user's query?
  4. retrieval_precision  – What fraction of retrieved chunks are above the relevance threshold?
  5. resume_grounding     – (interview) Are STAR answers grounded in the actual resume?

All metrics use Claude as the judge via structured JSON responses.
Run with `--eval` in main.py to include the report alongside the pipeline output.
"""

from __future__ import annotations

import json
import os
import textwrap
from dataclasses import dataclass, field
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# ── Thresholds ──────────────────────────────────────────────────────────────────
THRESHOLDS = {
    "context_relevance":   0.50,
    "faithfulness":        0.70,
    "answer_relevance":    0.70,
    "retrieval_precision": 0.50,
    "resume_grounding":    0.75,
}

_JUDGE_MODEL = "claude-haiku-4-5-20251001"   # fast + cheap for judge calls


# ── Result type ─────────────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    metric: str
    score: float          # 0.0 – 1.0
    threshold: float
    reasoning: str
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.score >= self.threshold

    @property
    def label(self) -> str:
        if self.score >= self.threshold:
            return "PASS"
        elif self.score >= self.threshold * 0.75:
            return "WARN"
        return "FAIL"


# ── Judge LLM helper ────────────────────────────────────────────────────────────

def _judge_llm() -> ChatAnthropic:
    return ChatAnthropic(
        model=_JUDGE_MODEL,
        anthropic_api_key=os.environ["ANTHROPIC_API_KEY"],
        temperature=0.0,
        max_tokens=2048,
    )


def _call_judge(llm: ChatAnthropic, prompt: str) -> dict:
    """
    Call the judge LLM and parse the JSON response.
    Returns an empty dict on any parse failure.
    """
    try:
        response = llm.invoke(prompt)
        text = response.content.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text)
    except (json.JSONDecodeError, Exception):
        return {}


# ── Metric 1: Context Relevance ─────────────────────────────────────────────────

_CONTEXT_RELEVANCE_PROMPT = """\
You are an evaluation judge for a Spotify content discovery RAG system.
The retrieved passages are Spotify metadata: show titles, publisher names,
episode counts, and show descriptions. This is the expected format — do not
penalise passages for being descriptive rather than explanatory.

Your task: assess whether each passage is relevant to the user's query,
given that the passages are Spotify catalogue metadata.

QUERY:
{query}

RETRIEVED PASSAGES:
{passages}

Scoring rubric for Spotify metadata passages:
  1.0 = Passage is about a show/book directly on this topic; description clearly matches the query
  0.7 = Passage is about a show/book related to this topic; partial match to the query
  0.4 = Passage mentions the topic incidentally or is only loosely related
  0.0 = Passage is about a completely different topic

For EACH passage (numbered 1..N), apply this rubric and assign a score.

Respond ONLY with valid JSON — no prose before or after:
{{
  "passage_scores": [<float>, ...],
  "mean_relevance": <float>,
  "reasoning": "<one sentence explaining the overall relevance of the retrieved set>"
}}"""


def context_relevance(
    query: str,
    retrieved_docs: list[tuple[Document, float]],
    llm: ChatAnthropic | None = None,
) -> EvalResult:
    """
    Rate how relevant the retrieved chunks are to the query.

    Uses an LLM judge to score each passage 0–1, then averages them.
    Falls back to FAISS similarity scores (converted from L2 distance) if
    the LLM call fails.

    Args:
        query:          The user's original query string.
        retrieved_docs: List of (Document, faiss_l2_score) tuples.
        llm:            Optional pre-built judge LLM (created if None).
    """
    llm = llm or _judge_llm()
    metric = "context_relevance"
    threshold = THRESHOLDS[metric]

    if not retrieved_docs:
        return EvalResult(
            metric=metric, score=0.0, threshold=threshold,
            reasoning="No documents were retrieved."
        )

    passages_text = "\n\n".join(
        f"[{i+1}] {doc.page_content[:300]}"
        for i, (doc, _) in enumerate(retrieved_docs)
    )
    prompt = _CONTEXT_RELEVANCE_PROMPT.format(query=query, passages=passages_text)
    result = _call_judge(llm, prompt)

    if result and "mean_relevance" in result:
        score = float(result["mean_relevance"])
        reasoning = result.get("reasoning", "")
        details = {"passage_scores": result.get("passage_scores", [])}
    else:
        # Fallback: use FAISS L2 distances converted to similarity pct
        sims = [max(0.0, 1.0 - score / 2.0) for _, score in retrieved_docs]
        score = sum(sims) / len(sims)
        reasoning = "LLM judge unavailable; score derived from FAISS similarity distances."
        details = {"fallback": True, "faiss_similarities": [round(s, 3) for s in sims]}

    return EvalResult(
        metric=metric, score=round(score, 3), threshold=threshold,
        reasoning=reasoning, details=details,
    )


# ── Metric 2: Faithfulness ───────────────────────────────────────────────────────

_FAITHFULNESS_PROMPT = """\
You are an evaluation judge assessing whether a generated answer is faithful
to (i.e. grounded in) the provided context. Faithfulness means every factual
claim in the answer can be verified from the context — no hallucinations.

CONTEXT (retrieved passages):
{context}

GENERATED ANSWER:
{answer}

Steps:
1. List the key factual claims made in the answer (max 10).
2. For each claim, mark it GROUNDED (supported by context) or NOT_GROUNDED.
3. Compute: faithfulness_score = grounded_count / total_claims

Respond ONLY with valid JSON:
{{
  "total_claims": <int>,
  "grounded_count": <int>,
  "faithfulness_score": <float 0-1>,
  "not_grounded_examples": ["<claim>", ...],
  "reasoning": "<one sentence>"
}}"""


def faithfulness(
    context: str,
    answer: str,
    llm: ChatAnthropic | None = None,
) -> EvalResult:
    """
    Check whether the generated answer is grounded in the retrieved context.

    Args:
        context: The formatted retrieved context string passed to the LLM.
        answer:  The generated report / answer text.
        llm:     Optional pre-built judge LLM.
    """
    llm = llm or _judge_llm()
    metric = "faithfulness"
    threshold = THRESHOLDS[metric]

    # Truncate to stay within context limits for the judge
    ctx_snippet = context[:3000] if len(context) > 3000 else context
    ans_snippet = answer[:2000] if len(answer) > 2000 else answer

    prompt = _FAITHFULNESS_PROMPT.format(context=ctx_snippet, answer=ans_snippet)
    result = _call_judge(llm, prompt)

    if result and "faithfulness_score" in result:
        score = float(result["faithfulness_score"])
        reasoning = result.get("reasoning", "")
        details = {
            "total_claims": result.get("total_claims", 0),
            "grounded_count": result.get("grounded_count", 0),
            "not_grounded_examples": result.get("not_grounded_examples", []),
        }
    else:
        score = 0.0
        reasoning = "LLM judge call failed; faithfulness could not be evaluated."
        details = {"error": True}

    return EvalResult(
        metric=metric, score=round(score, 3), threshold=threshold,
        reasoning=reasoning, details=details,
    )


# ── Metric 3: Answer Relevance ───────────────────────────────────────────────────

_ANSWER_RELEVANCE_PROMPT = """\
You are an evaluation judge assessing whether a generated answer addresses
the user's original query.

QUERY:
{query}

GENERATED ANSWER:
{answer}

Scoring rubric — read the full answer before scoring:
  1.0 = Opens by directly answering the query; every section ties back to it;
        recommendations are specific to what was asked.
  0.8 = Answer clearly addresses the query throughout; minor sections feel generic
        but the overall response is on-point.
  0.6 = Answer covers the query topic but sections are structured generically
        (e.g. a topic report) rather than framed as a direct answer.
  0.4 = Answer mentions the query topic but doesn't actually answer it — reads
        more like a catalogue than a response.
  0.2 = Answer is tangentially related; the query intent is not addressed.
  0.0 = Completely irrelevant.

Respond ONLY with valid JSON:
{{
  "score": <float 0-1>,
  "reasoning": "<two to three sentences explaining the score>"
}}"""


def answer_relevance(
    query: str,
    answer: str,
    llm: ChatAnthropic | None = None,
) -> EvalResult:
    """
    Check whether the generated answer actually addresses the user's query.

    Args:
        query:  The user's original query string.
        answer: The generated report / answer text.
        llm:    Optional pre-built judge LLM.
    """
    llm = llm or _judge_llm()
    metric = "answer_relevance"
    threshold = THRESHOLDS[metric]

    prompt = _ANSWER_RELEVANCE_PROMPT.format(
        query=query,
        answer=answer[:4000],
    )
    result = _call_judge(llm, prompt)

    if result and "score" in result:
        score = float(result["score"])
        reasoning = result.get("reasoning", "")
        details = {}
    else:
        score = 0.0
        reasoning = "LLM judge call failed; answer relevance could not be evaluated."
        details = {"error": True}

    return EvalResult(
        metric=metric, score=round(score, 3), threshold=threshold,
        reasoning=reasoning, details=details,
    )


# ── Metric 4: Retrieval Precision ────────────────────────────────────────────────

def retrieval_precision(
    retrieved_docs: list[tuple[Document, float]],
    similarity_threshold: float = 0.30,
    scores_are_similarities: bool = False,
) -> EvalResult:
    """
    Quality of the retrieved chunk set — no LLM call required.

    For FAISS L2 scores: uses an absolute threshold (scores converted to 0-1 similarity).

    For Cohere rerank scores: uses a distribution-aware check instead of an absolute
    threshold, because Cohere scores are relative — a score of 0.08 can still be the
    most relevant document in the corpus. Two signals are combined:
      - Drop-off ratio: how sharply do scores fall from best to worst?
        A tight cluster near the top suggests a focused, precise retrieval set.
      - Mean-relative precision: fraction of chunks scoring above 50% of the mean,
        identifying chunks that are meaningfully better than the average retrieval.
    """
    metric = "retrieval_precision"
    threshold = THRESHOLDS[metric]

    if not retrieved_docs:
        return EvalResult(
            metric=metric, score=0.0, threshold=threshold,
            reasoning="No documents retrieved."
        )

    if scores_are_similarities:
        raw_scores = [float(s) for _, s in retrieved_docs]
        score_source = "Cohere rerank"

        mean_score = sum(raw_scores) / len(raw_scores)
        top_score = max(raw_scores)
        bottom_score = min(raw_scores)

        # Signal 1: fraction above half the mean (relative relevance floor)
        rel_threshold = mean_score * 0.5
        above_rel = [s for s in raw_scores if s >= rel_threshold]
        mean_relative_precision = len(above_rel) / len(raw_scores)

        # Signal 2: score concentration — how much of the total score mass
        # sits in the top half of retrieved docs?
        sorted_scores = sorted(raw_scores, reverse=True)
        top_half = sorted_scores[: max(1, len(sorted_scores) // 2)]
        concentration = sum(top_half) / sum(raw_scores) if sum(raw_scores) > 0 else 0.5

        # Combine: weight concentration 60%, mean-relative precision 40%
        score = round(0.6 * concentration + 0.4 * mean_relative_precision, 3)

        reasoning = (
            f"Cohere scores — top: {top_score:.3f}, mean: {mean_score:.3f}, "
            f"bottom: {bottom_score:.3f}. "
            f"Score concentration (top half): {concentration:.2f}. "
            f"Mean-relative precision: {mean_relative_precision:.2f}."
        )
        details = {
            "scores": [round(s, 3) for s in raw_scores],
            "mean": round(mean_score, 3),
            "concentration": round(concentration, 3),
            "mean_relative_precision": round(mean_relative_precision, 3),
            "score_source": score_source,
        }
    else:
        similarities = [max(0.0, 1.0 - raw_score / 2.0) for _, raw_score in retrieved_docs]
        score_source = "FAISS L2"
        precise = [s for s in similarities if s >= similarity_threshold]
        score = round(len(precise) / len(similarities), 3)
        reasoning = (
            f"{len(precise)}/{len(similarities)} chunks above "
            f"{similarity_threshold * 100:.0f}% threshold ({score_source}). "
            f"Average: {sum(similarities)/len(similarities)*100:.1f}%."
        )
        details = {
            "similarities": [round(s, 3) for s in similarities],
            "threshold_used": similarity_threshold,
            "score_source": score_source,
        }

    return EvalResult(
        metric=metric, score=score, threshold=threshold,
        reasoning=reasoning, details=details,
    )


# ── Metric 5: Resume Grounding (interview-specific) ──────────────────────────────

_RESUME_GROUNDING_PROMPT = """\
You are an evaluation judge assessing whether suggested interview STAR answers
are grounded in the candidate's actual resume. A STAR answer is "grounded" when
every project, metric, skill, or achievement it mentions can be traced back to
the resume excerpts.

RESUME EXCERPTS:
{resume_context}

SUGGESTED STAR ANSWERS:
{star_answers}

Steps:
1. Identify all concrete claims (projects, metrics, tools, achievements) made in the STAR answers.
2. For each claim, determine if it appears in the resume excerpts.
3. Compute: grounding_score = grounded_claims / total_claims

Respond ONLY with valid JSON:
{{
  "total_claims": <int>,
  "grounded_claims": <int>,
  "grounding_score": <float 0-1>,
  "fabricated_examples": ["<fabricated claim>", ...],
  "reasoning": "<one sentence>"
}}"""


def resume_grounding(
    resume_context: str,
    star_answers: str,
    llm: ChatAnthropic | None = None,
) -> EvalResult:
    """
    Check whether STAR answers are grounded in the candidate's resume (no fabrication).

    Args:
        resume_context: Retrieved resume excerpts used to generate the answers.
        star_answers:   The generated STAR answers text.
        llm:            Optional pre-built judge LLM.
    """
    llm = llm or _judge_llm()
    metric = "resume_grounding"
    threshold = THRESHOLDS[metric]

    ctx_snippet = resume_context[:2500] if len(resume_context) > 2500 else resume_context
    ans_snippet = star_answers[:2000] if len(star_answers) > 2000 else star_answers

    prompt = _RESUME_GROUNDING_PROMPT.format(
        resume_context=ctx_snippet,
        star_answers=ans_snippet,
    )
    result = _call_judge(llm, prompt)

    if result and "grounding_score" in result:
        score = float(result["grounding_score"])
        reasoning = result.get("reasoning", "")
        details = {
            "total_claims": result.get("total_claims", 0),
            "grounded_claims": result.get("grounded_claims", 0),
            "fabricated_examples": result.get("fabricated_examples", []),
        }
    else:
        score = 0.0
        reasoning = "LLM judge call failed; resume grounding could not be evaluated."
        details = {"error": True}

    return EvalResult(
        metric=metric, score=round(score, 3), threshold=threshold,
        reasoning=reasoning, details=details,
    )


# ── Evaluator class ──────────────────────────────────────────────────────────────

class RAGEvaluator:
    """
    Orchestrates all RAG eval metrics and renders a summary report.

    Usage (Spotify mode):
        evaluator = RAGEvaluator()
        results = evaluator.run_spotify_evals(
            query=query,
            retrieved_docs=docs_with_scores,
            context=context_str,
            answer=report,
        )
        print(evaluator.format_report(results))

    Usage (Interview mode):
        results = evaluator.run_interview_evals(
            retrieved_docs=docs_with_scores,
            context=context_str,
            resume_context=resume_ctx,
            star_answers=answers_raw,
        )
    """

    def __init__(self) -> None:
        self._llm = _judge_llm()

    def run_spotify_evals(
        self,
        query: str,
        retrieved_docs: list[tuple[Document, float]],
        context: str,
        answer: str,
        scores_are_similarities: bool = False,
    ) -> list[EvalResult]:
        """Run all four Spotify-mode evals and return results."""
        print("  [eval] Running RAG evaluations (judge: claude-haiku)...")
        results = []
        results.append(context_relevance(query, retrieved_docs, llm=self._llm))
        results.append(faithfulness(context, answer, llm=self._llm))
        results.append(answer_relevance(query, answer, llm=self._llm))
        results.append(retrieval_precision(retrieved_docs, scores_are_similarities=scores_are_similarities))
        return results

    def run_interview_evals(
        self,
        retrieved_docs: list[tuple[Document, float]],
        context: str,
        resume_context: str,
        star_answers: str,
        scores_are_similarities: bool = False,
    ) -> list[EvalResult]:
        """Run retrieval precision + resume grounding for interview mode."""
        print("  [eval] Running interview RAG evaluations (judge: claude-haiku)...")
        results = []
        results.append(retrieval_precision(retrieved_docs, scores_are_similarities=scores_are_similarities))
        results.append(resume_grounding(resume_context, star_answers, llm=self._llm))
        return results

    # ── Report formatter ────────────────────────────────────────────────────────

    @staticmethod
    def format_report(results: list[EvalResult]) -> str:
        """Render a tabular eval report as a printable string."""
        if not results:
            return "No eval results to display."

        col_metric    = 24
        col_score     = 8
        col_label     = 7
        col_threshold = 11
        width = 70  # total box width (excluding border chars)

        header = (
            f"{'Metric':<{col_metric}} "
            f"{'Score':>{col_score}} "
            f"{'Label':<{col_label}} "
            f"{'Threshold':>{col_threshold}}"
        )
        sep = "─" * width

        lines = [
            "",
            "╔" + "═" * width + "╗",
            "║  RAG Evaluation Report" + " " * (width - 23) + "║",
            "╠" + "═" * width + "╣",
            "║  " + header + " " * max(0, width - len(header) - 2) + "║",
            "║  " + sep + "║",
        ]

        for r in results:
            icon = "✓" if r.passed else "✗"
            row = (
                f"{r.metric:<{col_metric}} "
                f"{r.score:>{col_score}.3f} "
                f"{icon} {r.label:<{col_label-2}} "
                f"{r.threshold:>{col_threshold}.2f}"
            )
            lines.append("║  " + row + " " * max(0, width - len(row) - 2) + "║")

        lines.append("╚" + "═" * width + "╝")
        lines.append("")

        # Full reasoning for every metric, word-wrapped to 76 chars
        lines.append("  Reasoning")
        lines.append("  " + "─" * 68)
        for r in results:
            if r.reasoning:
                icon = "✓" if r.passed else "✗"
                lines.append(f"  {icon} {r.metric}:")
                for line in textwrap.wrap(r.reasoning, width=70):
                    lines.append(f"      {line}")
                lines.append("")

        # Fabrication warnings from resume_grounding
        for r in results:
            if r.metric == "resume_grounding" and r.details.get("fabricated_examples"):
                lines.append("  [eval] Possible fabricated claims in STAR answers:")
                for ex in r.details["fabricated_examples"]:
                    lines.append(f"    • {ex}")
                lines.append("")

        # Not-grounded warnings from faithfulness
        for r in results:
            if r.metric == "faithfulness" and r.details.get("not_grounded_examples"):
                lines.append("  [eval] Claims not grounded in context:")
                for ex in r.details["not_grounded_examples"]:
                    lines.append(f"    • {ex}")
                lines.append("")

        return "\n".join(lines)
