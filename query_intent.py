"""
query_intent.py
Classifies a Spotify search query into one of four intent types, then
returns the synthesis prompt template tailored to that intent.

Intent types
────────────
  discovery      — User wants to explore what's broadly available on a topic.
                   e.g. "mindfulness meditation", "personal finance podcasts"

  recommendation — User wants specific picks suited to their situation.
                   e.g. "best meditation podcast for beginners",
                        "recommend sleep podcasts for anxiety"

  comparison     — User wants to compare formats, shows, or approaches.
                   e.g. "podcast vs audiobook for stoicism",
                        "Huberman Lab vs Lex Fridman"

  deep_dive      — User wants thorough thematic analysis of a specific topic.
                   e.g. "everything about intermittent fasting",
                        "deep dive into behavioral economics"
"""

from __future__ import annotations

import json
import os

from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

_CLASSIFIER_MODEL = "claude-haiku-4-5-20251001"

# ── Classification prompt ────────────────────────────────────────────────────────

_CLASSIFY_PROMPT = """\
Classify the following Spotify search query into exactly one of four intent types.

QUERY: "{query}"

Intent definitions:
  discovery      — The user wants to explore what is broadly available on a topic.
                   Signals: bare topic names, "find", "show me", no specific outcome mentioned.
  recommendation — The user wants specific picks tailored to their situation or goal.
                   Signals: "best for", "recommend", "should I", audience qualifiers
                   (beginner, expert, anxious, busy...), desired outcome stated.
  comparison     — The user wants to compare formats, specific shows, or approaches.
                   Signals: "vs", "compared to", "difference between", "which is better",
                   two or more named subjects.
  deep_dive      — The user wants comprehensive, thematic analysis of a specific topic.
                   Signals: "everything about", "deep dive", "comprehensive", "all about",
                   highly specific niche topic.

Respond ONLY with valid JSON — no prose before or after:
{{
  "intent": "<discovery|recommendation|comparison|deep_dive>",
  "confidence": <float 0.0-1.0>,
  "reasoning": "<one sentence>",
  "sub_questions": [
    "<specific thing the user wants to know 1>",
    "<specific thing the user wants to know 2>",
    "<specific thing the user wants to know 3>"
  ]
}}"""


def classify_intent(query: str) -> dict:
    """
    Classify a query's intent using claude-haiku.

    Returns a dict with keys: intent, confidence, reasoning, sub_questions.
    Falls back to 'discovery' intent on any failure.
    """
    llm = ChatAnthropic(
        model=_CLASSIFIER_MODEL,
        anthropic_api_key=os.environ["ANTHROPIC_API_KEY"],
        temperature=0.0,
        max_tokens=512,
    )
    try:
        response = llm.invoke(_CLASSIFY_PROMPT.format(query=query))
        text = response.content.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        result = json.loads(text)
        if result.get("intent") not in {"discovery", "recommendation", "comparison", "deep_dive"}:
            result["intent"] = "discovery"
        return result
    except Exception:
        return {
            "intent": "discovery",
            "confidence": 0.0,
            "reasoning": "Classification failed; defaulting to discovery.",
            "sub_questions": [
                f"What podcasts are available about {query}?",
                f"What audiobooks are available about {query}?",
                f"What are the most relevant picks for {query}?",
            ],
        }


# ── Intent-specific synthesis prompt templates ───────────────────────────────────
# Each template receives: query, sub_questions, podcasts_section,
#                         audiobooks_section, retrieved_context

_GROUNDING_RULES = """\
STRICT GROUNDING RULES — follow these without exception:
- Every claim, description, or recommendation must be directly supported by the passages below.
- Do NOT add background knowledge or details not explicitly stated in the passages.
- Do NOT infer episode counts, author names, or audience descriptions unless verbatim in passages.
- If passages lack enough information for a section, write only what they support."""

# ── Discovery ──────────────────────────────────────────────────────────────────

_DISCOVERY_PROMPT = """You are an expert Spotify content analyst.

A user searched for: "{query}"
They want to explore what's available on this topic.

{grounding_rules}

══════════════════════════════════════════════
RANKED PODCASTS
══════════════════════════════════════════════
{podcasts_section}

══════════════════════════════════════════════
RANKED AUDIOBOOKS
══════════════════════════════════════════════
{audiobooks_section}

══════════════════════════════════════════════
MOST RELEVANT CONTENT
══════════════════════════════════════════════
{retrieved_context}

══════════════════════════════════════════════
The user specifically wants answers to:
{sub_questions}

Write a response that directly answers the query. Do NOT output the sub-questions
or any decomposition step — use them internally to shape your sections.

Structure:
1. Open with one paragraph directly summarising what Spotify offers for "{query}".
2. One section per sub-question above — title each section with a short answer-oriented
   heading (not the sub-question verbatim).
3. Close with "Where to Start": top 2-3 picks from the ranked results with a one-line
   reason each, grounded in the passages.
"""

# ── Recommendation ─────────────────────────────────────────────────────────────

_RECOMMENDATION_PROMPT = """You are an expert Spotify content analyst.

A user asked: "{query}"
They want specific recommendations tailored to their situation.

{grounding_rules}

══════════════════════════════════════════════
RANKED PODCASTS
══════════════════════════════════════════════
{podcasts_section}

══════════════════════════════════════════════
RANKED AUDIOBOOKS
══════════════════════════════════════════════
{audiobooks_section}

══════════════════════════════════════════════
MOST RELEVANT CONTENT
══════════════════════════════════════════════
{retrieved_context}

══════════════════════════════════════════════
Internally use these sub-questions to guide your answer — do NOT print them:
{sub_questions}

Structure (lead with the answer, never with an overview):
1. Open with: "For [restate the query], the best option on Spotify is [X] because [reason from passages]."
2. One section per sub-question — use short answer-oriented headings, not the sub-question text.
   For each pick, explain why it fits the user's specific situation using evidence from the passages.
3. Close with "Top 3 Picks": ranked shortlist with one justification line each.
"""

# ── Comparison ─────────────────────────────────────────────────────────────────

_COMPARISON_PROMPT = """You are an expert Spotify content analyst.

A user asked: "{query}"
They want to compare formats, shows, or approaches.

{grounding_rules}

══════════════════════════════════════════════
RANKED PODCASTS
══════════════════════════════════════════════
{podcasts_section}

══════════════════════════════════════════════
RANKED AUDIOBOOKS
══════════════════════════════════════════════
{audiobooks_section}

══════════════════════════════════════════════
MOST RELEVANT CONTENT
══════════════════════════════════════════════
{retrieved_context}

══════════════════════════════════════════════
Internally use these sub-questions to guide your answer — do NOT print them:
{sub_questions}

Structure (open with the verdict, not with background):
1. Open with a direct verdict: which option better serves the user's goal and why,
   grounded strictly in the passages.
2. One comparison section per sub-question — use short answer-oriented headings.
   Structure each as [Option A] vs [Option B] with key differences from the passages.
3. Close with "Our Verdict": clear recommendation with reasoning from the passages.
"""

# ── Deep Dive ──────────────────────────────────────────────────────────────────

_DEEP_DIVE_PROMPT = """You are an expert Spotify content analyst.

A user asked: "{query}"
They want a thorough, thematic analysis of this topic.

{grounding_rules}

══════════════════════════════════════════════
RANKED PODCASTS
══════════════════════════════════════════════
{podcasts_section}

══════════════════════════════════════════════
RANKED AUDIOBOOKS
══════════════════════════════════════════════
{audiobooks_section}

══════════════════════════════════════════════
MOST RELEVANT CONTENT
══════════════════════════════════════════════
{retrieved_context}

══════════════════════════════════════════════
Internally use these sub-questions to guide your analysis — do NOT print them:
{sub_questions}

Structure (open with insight, not with a list of shows):
1. Open with 2-3 sentences synthesising what the Spotify content collectively reveals
   about "{query}" — lead with the most interesting insight from the passages.
2. One thematic section per sub-question — use insight-oriented headings, not the
   sub-question text. Draw connections across multiple shows where passages support it.
3. Close with "Key Takeaways": 3-4 most important insights that emerge from the
   passages as a whole.
"""

_INTENT_PROMPTS = {
    "discovery":     _DISCOVERY_PROMPT,
    "recommendation": _RECOMMENDATION_PROMPT,
    "comparison":    _COMPARISON_PROMPT,
    "deep_dive":     _DEEP_DIVE_PROMPT,
}


def build_synthesis_prompt(
    intent_result: dict,
    query: str,
    podcasts_section: str,
    audiobooks_section: str,
    retrieved_context: str,
) -> str:
    """
    Select the intent-specific prompt template and format it with all pipeline outputs.

    Args:
        intent_result:     Output of classify_intent() — must have 'intent' and 'sub_questions'.
        query:             The user's original query string.
        podcasts_section:  Formatted podcast results text.
        audiobooks_section: Formatted audiobook results text.
        retrieved_context: Retrieved and reranked context string.

    Returns:
        A fully formatted synthesis prompt ready to send to the LLM.
    """
    intent = intent_result.get("intent", "discovery")
    sub_questions = intent_result.get("sub_questions", [])
    template = _INTENT_PROMPTS.get(intent, _DISCOVERY_PROMPT)

    sub_q_formatted = "\n".join(f"  {i+1}. {q}" for i, q in enumerate(sub_questions))

    return template.format(
        query=query,
        grounding_rules=_GROUNDING_RULES,
        podcasts_section=podcasts_section,
        audiobooks_section=audiobooks_section,
        retrieved_context=retrieved_context,
        sub_questions=sub_q_formatted,
    )
