"""
agent.py
Orchestrates the Spotify RAG pipeline:
  1. Input guardrails (query validation, prompt injection check)
  2. Spotify API search — top-10 podcasts + top-3 audiobooks
  3. Cohere rerank + composite scoring (ranking.py)
  4. Cohere embeddings + FAISS vector store (embeddings_store.py)
  5. Two-stage retrieval — FAISS candidate pool → Cohere rerank (k=8)
  6. Query intent classification — discovery / recommendation / comparison / deep_dive
  7. Intent-aware synthesis via Claude using intent-specific prompt templates
  8. Output guardrails (length and completeness checks)
"""

import os
from typing import Any
from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate

from spotify_tools import search_spotify_podcasts, search_spotify_audiobooks
from embeddings_store import build_vector_store, retrieve_and_rerank
from ranking import rank_results
from guardrails import validate_query, check_retrieval_quality, check_output_completeness
from query_intent import classify_intent, build_synthesis_prompt

load_dotenv()

# ── LangChain tools available to the agent ─────────────────────────────────────
TOOLS = [search_spotify_podcasts, search_spotify_audiobooks]

# ── ReAct prompt template ───────────────────────────────────────────────────────
REACT_PROMPT = PromptTemplate.from_template(
    """You are a research assistant that finds and analyses Spotify content.

You have access to the following tools:

{tools}

Use the following format strictly:

Question: the input question you must answer
Thought: think about what to do
Action: the action to take, must be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Important rules:
- Always call BOTH search_spotify_podcasts AND search_spotify_audiobooks for every query.
- Base your Final Answer only on what you observed from the tools.
- Do NOT make up podcast or audiobook titles.

Begin!

Question: {input}
Thought: {agent_scratchpad}"""
)


def _format_podcast_section(podcasts: list[dict[str, Any]]) -> str:
    """Format a ranked list of podcast dicts into a human-readable text block for the synthesis prompt."""
    lines = []
    for p in podcasts:
        score = p.get("relevance_score", "N/A")
        lines.append(
            f"  #{p['rank']} [{score}] {p['name']} – {p['publisher']}\n"
            f"      Episodes: {p['total_episodes']} | {p['external_url']}\n"
            f"      {p['description'][:200]}..."
        )
    return "\n\n".join(lines) if lines else "No podcasts found."


def _format_audiobook_section(audiobooks: list[dict[str, Any]]) -> str:
    """Format a ranked list of audiobook dicts into a human-readable text block for the synthesis prompt."""
    lines = []
    for ab in audiobooks:
        score = ab.get("relevance_score", "N/A")
        authors = ", ".join(ab.get("authors", [])) or "Unknown"
        lines.append(
            f"  #{ab['rank']} [{score}] {ab['name']} – by {authors}\n"
            f"      Chapters: {ab['total_chapters']} | {ab['external_url']}\n"
            f"      {ab['description'][:200]}..."
        )
    return "\n\n".join(lines) if lines else "No audiobooks found."


class SpotifyRAGAgent:
    """
    High-level orchestrator that runs the full RAG pipeline:
    search → rank → embed → retrieve → synthesise.
    """

    def __init__(self) -> None:
        self.llm = ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
            anthropic_api_key=os.environ["ANTHROPIC_API_KEY"],
            temperature=0.3,
            max_tokens=4096,
        )
        react_agent = create_react_agent(llm=self.llm, tools=TOOLS, prompt=REACT_PROMPT)
        self.agent_executor = AgentExecutor(
            agent=react_agent,
            tools=TOOLS,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=6,
        )

    def run(self, query: str, run_evals: bool = False) -> str:
        """
        Execute the full pipeline for a user query and return the final report.

        Args:
            query:     Natural-language question / topic (e.g. "mindfulness meditation").
            run_evals: If True, run LLM-as-judge RAG evaluations and append the
                       eval report to stdout after the pipeline completes.

        Returns:
            A comprehensive markdown-formatted report string.
        """
        print(f"\n{'='*60}")
        print(f"  Query: {query}")
        print(f"{'='*60}\n")

        # ── Input guardrail ────────────────────────────────────────────────────
        guard = validate_query(query)
        guard.print_warnings()
        guard.raise_if_failed()

        # ── Step 1: Retrieve via Spotify tools ─────────────────────────────────
        print("► Step 1: Searching Spotify...")
        podcasts_raw: list[dict] = search_spotify_podcasts.invoke(query)
        audiobooks_raw: list[dict] = search_spotify_audiobooks.invoke(query)
        print(f"  Found {len(podcasts_raw)} podcasts, {len(audiobooks_raw)} audiobooks.")

        # ── Step 2: Rank results ────────────────────────────────────────────────
        print("► Step 2: Ranking results with Cohere rerank...")
        ranked = rank_results(query, podcasts_raw, audiobooks_raw)
        podcasts = ranked["podcasts"]
        audiobooks = ranked["audiobooks"]
        print(f"  Ranking complete.")

        # ── Step 3: Build FAISS vector store ───────────────────────────────────
        print("► Step 3: Building FAISS vector store with Cohere embeddings...")
        vector_store = build_vector_store(podcasts, audiobooks)
        print(f"  Vector store built with {len(podcasts) + len(audiobooks)} documents.")

        # ── Step 4: Retrieve most relevant context (FAISS → Cohere rerank) ───────
        print("► Step 4: Retrieving and reranking top chunks with Cohere...")
        context, docs_with_scores = retrieve_and_rerank(vector_store, query, k=8)

        # Retrieval quality guardrail — pass raw Cohere scores (0-1)
        cohere_scores = [score for _, score in docs_with_scores]
        retrieval_guard = check_retrieval_quality(cohere_scores, scores_are_cohere=True)
        retrieval_guard.print_warnings()

        # ── Step 5: Classify query intent ──────────────────────────────────────
        print("► Step 5: Classifying query intent...")
        intent_result = classify_intent(query)
        print(
            f"  Intent: {intent_result['intent']} "
            f"(confidence: {intent_result.get('confidence', '?'):.2f}) — "
            f"{intent_result.get('reasoning', '')}"
        )

        # ── Step 6: Synthesise with Claude using intent-specific prompt ─────────
        print("► Step 6: Synthesising report with Claude...")
        synthesis_input = build_synthesis_prompt(
            intent_result=intent_result,
            query=query,
            podcasts_section=_format_podcast_section(podcasts),
            audiobooks_section=_format_audiobook_section(audiobooks),
            retrieved_context=context,
        )
        response = self.llm.invoke(synthesis_input)
        report: str = response.content

        # Output guardrail — sections are now query-driven so only check length
        output_guard = check_output_completeness(report)
        output_guard.print_warnings()
        output_guard.raise_if_failed()

        print("\n► Done!\n")

        # ── Optional: RAG evaluations ───────────────────────────────────────────
        if run_evals:
            from rag_evals import RAGEvaluator
            evaluator = RAGEvaluator()
            eval_results = evaluator.run_spotify_evals(
                query=query,
                retrieved_docs=docs_with_scores,
                context=context,
                answer=report,
                scores_are_similarities=True,
            )
            print(evaluator.format_report(eval_results))

        return report
