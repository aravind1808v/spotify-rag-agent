"""
agent.py
LangChain ReAct agent that:
  1. Calls the Spotify tools to fetch top-10 podcasts + top-3 audiobooks
  2. Ranks results using Cohere rerank (ranking.py)
  3. Embeds all content into a FAISS vector store (embeddings_store.py)
  4. Retrieves the most semantically relevant chunks
  5. Synthesises a comprehensive report using Claude (Anthropic)
"""

import os
from typing import Any
from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate

from spotify_tools import search_spotify_podcasts, search_spotify_audiobooks
from embeddings_store import build_vector_store, retrieve_relevant_context
from ranking import rank_results

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

# ── Synthesis prompt ────────────────────────────────────────────────────────────
SYNTHESIS_PROMPT = """You are an expert content analyst. A user asked:

"{query}"

Below are the top Spotify podcasts and audiobooks on this topic, ranked by relevance.
Use the retrieved context to write a comprehensive, well-structured report.

══════════════════════════════════════════════
RANKED PODCASTS (Top 10)
══════════════════════════════════════════════
{podcasts_section}

══════════════════════════════════════════════
RANKED AUDIOBOOKS (Top 3)
══════════════════════════════════════════════
{audiobooks_section}

══════════════════════════════════════════════
MOST RELEVANT CONTENT (semantic search)
══════════════════════════════════════════════
{retrieved_context}

══════════════════════════════════════════════
Write a comprehensive report covering:
1. **Overview** – What is this topic about and why is it popular on Spotify?
2. **Top Podcasts** – Highlight the best podcasts, what makes each unique, and who they're for.
3. **Top Audiobooks** – Summarise each audiobook, its authors, and key themes.
4. **Key Themes** – Common threads across the podcasts and audiobooks.
5. **Recommendations** – Your top picks based on depth, popularity, and relevance.

Format the report with clear headings and bullet points where appropriate.
"""


def _format_podcast_section(podcasts: list[dict[str, Any]]) -> str:
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

    def run(self, query: str) -> str:
        """
        Execute the full pipeline for a user query and return the final report.

        Args:
            query: Natural-language question / topic (e.g. "mindfulness meditation").

        Returns:
            A comprehensive markdown-formatted report string.
        """
        print(f"\n{'='*60}")
        print(f"  Query: {query}")
        print(f"{'='*60}\n")

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

        # ── Step 4: Retrieve most relevant context ─────────────────────────────
        print("► Step 4: Retrieving top relevant chunks...")
        context = retrieve_relevant_context(vector_store, query, k=6)

        # ── Step 5: Synthesise with Claude ─────────────────────────────────────
        print("► Step 5: Synthesising report with Claude...")
        synthesis_input = SYNTHESIS_PROMPT.format(
            query=query,
            podcasts_section=_format_podcast_section(podcasts),
            audiobooks_section=_format_audiobook_section(audiobooks),
            retrieved_context=context,
        )
        response = self.llm.invoke(synthesis_input)
        report: str = response.content

        print("\n► Done!\n")
        return report
