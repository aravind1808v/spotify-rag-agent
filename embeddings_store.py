"""
embeddings_store.py
Builds a FAISS vector store from Spotify search results using Cohere embeddings.
Provides semantic search so the agent can retrieve the most relevant chunks
when synthesising its final answer.
"""

import os
from typing import Any
import cohere
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langsmith import traceable
from dotenv import load_dotenv

load_dotenv()


def _results_to_documents(
    podcasts: list[dict[str, Any]],
    audiobooks: list[dict[str, Any]],
) -> list[Document]:
    """
    Convert raw Spotify result dicts into LangChain Document objects.
    Each document's page_content is a human-readable text block that
    Cohere will embed; metadata preserves all structured fields for display.
    """
    docs: list[Document] = []

    for p in podcasts:
        content = (
            f"[PODCAST – Rank {p['rank']}]\n"
            f"Title: {p['name']}\n"
            f"Publisher: {p['publisher']}\n"
            f"Episodes: {p['total_episodes']}\n"
            f"Description: {p['description']}\n"
            f"URL: {p['external_url']}"
        )
        docs.append(Document(page_content=content, metadata={**p}))

    for ab in audiobooks:
        authors_str = ", ".join(ab["authors"]) if ab["authors"] else "Unknown"
        narrators_str = ", ".join(ab["narrators"]) if ab["narrators"] else "Unknown"
        content = (
            f"[AUDIOBOOK – Rank {ab['rank']}]\n"
            f"Title: {ab['name']}\n"
            f"Authors: {authors_str}\n"
            f"Narrators: {narrators_str}\n"
            f"Chapters: {ab['total_chapters']}\n"
            f"Description: {ab['description']}\n"
            f"URL: {ab['external_url']}"
        )
        docs.append(Document(page_content=content, metadata={**ab}))

    return docs


def build_vector_store(
    podcasts: list[dict[str, Any]],
    audiobooks: list[dict[str, Any]],
) -> FAISS:
    """
    Embed all retrieved Spotify results with Cohere and index them in FAISS.

    Args:
        podcasts:   List of podcast dicts from search_spotify_podcasts tool.
        audiobooks: List of audiobook dicts from search_spotify_audiobooks tool.

    Returns:
        A FAISS vector store ready for similarity search.
    """
    embedding_model = CohereEmbeddings(
        model=os.getenv("COHERE_EMBEDDING_MODEL", "embed-english-v3.0"),
        cohere_api_key=os.environ["COHERE_API_KEY"],
    )
    docs = _results_to_documents(podcasts, audiobooks)
    if not docs:
        raise ValueError("No documents to embed – search returned empty results.")
    return FAISS.from_documents(docs, embedding_model)


def build_vector_store_from_docs(docs: list[Document]) -> FAISS:
    """
    Embed arbitrary pre-built Document objects with Cohere and index in FAISS.
    Used by the interview prep pipeline where documents are already constructed
    from resume and JD chunks (rather than from Spotify result dicts).

    Args:
        docs: List of LangChain Document objects to embed and index.

    Returns:
        A FAISS vector store ready for similarity search.
    """
    embedding_model = CohereEmbeddings(
        model=os.getenv("COHERE_EMBEDDING_MODEL", "embed-english-v3.0"),
        cohere_api_key=os.environ["COHERE_API_KEY"],
    )
    if not docs:
        raise ValueError("No documents to embed.")
    return FAISS.from_documents(docs, embedding_model)


@traceable(name="retrieve_and_rerank")
def retrieve_and_rerank(
    vector_store: FAISS,
    query: str,
    k: int = 8,
    fetch_k: int | None = None,
    source_filter: str | None = None,
) -> tuple[str, list[tuple[Document, float]]]:
    """
    Two-stage retrieval: broad FAISS candidate fetch → Cohere rerank.

    Stage 1 — FAISS retrieves `fetch_k` candidates quickly (default k×3).
    Stage 2 — Cohere rerank re-scores those candidates using a cross-encoder
               that jointly reads the query and each document, then returns the
               top `k` by relevance.

    Args:
        vector_store:  The FAISS index to search.
        query:         The user's query string.
        k:             Number of top documents to return after reranking.
        fetch_k:       Candidate pool size for FAISS (defaults to k × 3).
        source_filter: If set, keep only documents whose metadata["source"]
                       matches this value (e.g. "resume" or "jd").

    Returns:
        A tuple of:
          - formatted context string (ready to inject into a prompt)
          - list of (Document, cohere_relevance_score) where score is in [0, 1]
    """
    fetch_k = fetch_k or k * 3

    # Stage 1: broad FAISS retrieval
    candidates = vector_store.similarity_search(query, k=fetch_k)

    if source_filter:
        candidates = [d for d in candidates if d.metadata.get("source") == source_filter]

    if not candidates:
        label = f"'{source_filter}'" if source_filter else "any"
        return f"No {label} content retrieved.", []

    # Stage 2: Cohere rerank
    co = cohere.Client(os.environ["COHERE_API_KEY"])
    texts = [doc.page_content for doc in candidates]
    response = co.rerank(
        model=os.getenv("COHERE_RERANK_MODEL", "rerank-english-v3.0"),
        query=query,
        documents=texts,
        top_n=min(k, len(candidates)),
    )

    reranked: list[tuple[Document, float]] = [
        (candidates[r.index], float(r.relevance_score))
        for r in response.results
    ]

    context_parts = [
        f"--- [Rerank Score: {score:.3f}] ---\n{doc.page_content}"
        for doc, score in reranked
    ]
    return "\n\n".join(context_parts), reranked


def retrieve_with_scores(
    vector_store: FAISS,
    query: str,
    k: int = 6,
) -> list[tuple[Document, float]]:
    """
    Return the top-k most relevant documents together with their raw FAISS L2
    scores. Used by the RAG evaluators which need the raw scores, not just the
    formatted context string.

    Args:
        vector_store: The FAISS index.
        query:        The user's original question / topic.
        k:            Number of top documents to retrieve (default 6).

    Returns:
        List of (Document, l2_score) tuples sorted by ascending L2 distance
        (i.e. descending relevance).
    """
    return vector_store.similarity_search_with_score(query, k=k)


def retrieve_relevant_context(
    vector_store: FAISS,
    query: str,
    k: int = 6,
) -> str:
    """
    Retrieve the top-k most relevant documents from the FAISS store and
    format them as a single context string for the LLM.

    Args:
        vector_store: The FAISS index built from Spotify results.
        query:        The user's original question / topic.
        k:            Number of top documents to retrieve (default 6).

    Returns:
        A formatted string containing the retrieved context passages.
    """
    results = vector_store.similarity_search_with_score(query, k=k)
    context_parts = []
    for doc, score in results:
        # Lower L2 distance = more similar; convert to a similarity % for display
        similarity_pct = max(0.0, 1.0 - score / 2.0) * 100
        context_parts.append(
            f"--- [Relevance: {similarity_pct:.1f}%] ---\n{doc.page_content}"
        )
    return "\n\n".join(context_parts)
