"""
ranking.py
Ranks Spotify search results by a composite relevance score that combines:
  1. Cohere semantic similarity (embedding dot-product) to the user query
  2. Popularity signals (episode count for podcasts, chapter count for audiobooks)
  3. Description length (a proxy for content richness)

The final score is normalised to [0, 1] for each signal, then combined with
configurable weights.
"""

from __future__ import annotations

import os
import math
from typing import Any
import cohere
from dotenv import load_dotenv

load_dotenv()

# ── Weighting knobs ─────────────────────────────────────────────────────────────
WEIGHT_SEMANTIC = 0.60   # relevance to the query (most important)
WEIGHT_POPULARITY = 0.25  # episode / chapter count
WEIGHT_RICHNESS = 0.15    # description length


def _cohere_similarities(query: str, texts: list[str]) -> list[float]:
    """
    Use the Cohere rerank endpoint to get relevance scores for each text
    relative to the query.  Returns a list of floats in [0, 1].
    """
    co = cohere.Client(os.environ["COHERE_API_KEY"])
    rerank_model = "rerank-english-v3.0"
    response = co.rerank(
        model=rerank_model,
        query=query,
        documents=texts,
        top_n=len(texts),
    )
    # Build a score list aligned with the original texts order
    scores: list[float] = [0.0] * len(texts)
    for result in response.results:
        scores[result.index] = float(result.relevance_score)
    return scores


def _normalise(values: list[float]) -> list[float]:
    """Min-max normalise a list of floats to [0, 1]."""
    if not values:
        return values
    min_v, max_v = min(values), max(values)
    if max_v == min_v:
        return [1.0] * len(values)
    return [(v - min_v) / (max_v - min_v) for v in values]


def rank_results(
    query: str,
    podcasts: list[dict[str, Any]],
    audiobooks: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """
    Rank podcasts and audiobooks separately, enriching each item with a
    ``relevance_score`` and updated ``rank`` field.

    Args:
        query:      The user's original search query.
        podcasts:   Raw podcast dicts from search_spotify_podcasts.
        audiobooks: Raw audiobook dicts from search_spotify_audiobooks.

    Returns:
        A dict with keys "podcasts" and "audiobooks", each a list sorted by
        descending relevance_score with rank fields updated accordingly.
    """

    def _score_items(items: list[dict[str, Any]], count_field: str) -> list[dict[str, Any]]:
        if not items:
            return []

        texts = [
            f"{item['name']}. {item.get('description', '')}" for item in items
        ]

        # 1. Semantic similarity via Cohere rerank
        semantic_scores = _cohere_similarities(query, texts)
        norm_semantic = _normalise(semantic_scores)

        # 2. Popularity signal (log-scaled so outliers don't dominate)
        counts = [math.log1p(item.get(count_field, 0)) for item in items]
        norm_counts = _normalise(counts)

        # 3. Description richness (character count, capped at 1000)
        richness = [min(len(item.get("description", "")), 1000) for item in items]
        norm_richness = _normalise([float(r) for r in richness])

        scored = []
        for i, item in enumerate(items):
            composite = (
                WEIGHT_SEMANTIC * norm_semantic[i]
                + WEIGHT_POPULARITY * norm_counts[i]
                + WEIGHT_RICHNESS * norm_richness[i]
            )
            scored.append({**item, "relevance_score": round(composite, 4)})

        scored.sort(key=lambda x: x["relevance_score"], reverse=True)
        for new_rank, item in enumerate(scored, start=1):
            item["rank"] = new_rank
        return scored

    ranked_podcasts = _score_items(podcasts, count_field="total_episodes")
    ranked_audiobooks = _score_items(audiobooks, count_field="total_chapters")

    return {"podcasts": ranked_podcasts, "audiobooks": ranked_audiobooks}
