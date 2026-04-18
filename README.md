# Spotify RAG Agent

A RAG (Retrieval-Augmented Generation) agent that combines the **Spotify API**, **Cohere**, and **Claude** to deliver two features:

1. **Spotify Content Explorer** — search any topic and get a ranked, synthesised report of the top podcasts and audiobooks on Spotify. Adapts its output structure to your query intent.
2. **Interview Prep** — upload your resume and a job description to get custom interview questions, STAR-method answers, and a personalised preparation strategy.

---

## Architecture

```
                        ┌─────────────────────────────────────────┐
                        │              main.py (CLI)               │
                        └────────────┬────────────┬───────────────┘
                                     │            │
               ┌─────────────────────▼──┐    ┌───▼──────────────────────┐
               │   Spotify RAG Mode     │    │   Interview Prep Mode    │
               │   (agent.py)           │    │   (interview_agent.py)   │
               └──────────┬─────────────┘    └───────────┬──────────────┘
                          │                              │
          ┌───────────────▼──────────────┐  ┌───────────▼──────────────┐
          │  1. Input Guardrails         │  │  1. Input Guardrails     │
          │     (guardrails.py)          │  │     (guardrails.py)      │
          ├──────────────────────────────┤  ├──────────────────────────┤
          │  2. Spotify API Search       │  │  2. Parse Resume + JD    │
          │     (spotify_tools.py)       │  │     (resume_parser.py)   │
          ├──────────────────────────────┤  ├──────────────────────────┤
          │  3. Cohere Rerank            │  │  3. Chunk + Tag by source│
          │     (ranking.py)             │  │     (resume_parser.py)   │
          ├──────────────────────────────┤  ├──────────────────────────┤
          │  4. Cohere Embeddings        │  │  4. Cohere Embeddings    │
          │  +  FAISS Vector Store       │  │  +  FAISS Vector Store   │
          │     (embeddings_store.py)    │  │     (embeddings_store.py)│
          ├──────────────────────────────┤  ├──────────────────────────┤
          │  5. FAISS → Cohere Rerank    │  │  5. FAISS → Cohere Rerank│
          │     (two-stage retrieval)    │  │     per source filter    │
          ├──────────────────────────────┤  ├──────────────────────────┤
          │  6. Query Intent Classify    │  │  6. Claude: Questions    │
          │     (query_intent.py)        │  │         + STAR Answers   │
          ├──────────────────────────────┤  │         + Prep Strategy  │
          │  7. Intent-aware Synthesis   │  ├──────────────────────────┤
          │     Claude (4 prompt shapes) │  │  7. Output Guardrails    │
          ├──────────────────────────────┤  │     (guardrails.py)      │
          │  8. Output Guardrails        │  └──────────────────────────┘
          │     (guardrails.py)          │
          └──────────────────────────────┘
```

---

## Tech Stack

| Component | Tool |
|---|---|
| LLM (synthesis) | Claude Sonnet via `langchain-anthropic` |
| LLM (judge + classifier) | Claude Haiku via `langchain-anthropic` |
| Embeddings | Cohere `embed-english-v3.0` |
| Reranking | Cohere `rerank-english-v3.0` |
| Vector Store | FAISS (in-memory) |
| Spotify Search | Spotipy (Spotify Web API) |
| PDF Parsing | pypdf |
| Web Scraping | LangChain `WebBaseLoader` + BeautifulSoup |
| Orchestration | LangChain |
| Observability | LangSmith |

---

## Setup

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/aravind1808v/spotify-rag-agent.git
cd spotify-rag-agent
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
```

Edit `.env`:

```env
SPOTIFY_CLIENT_ID=your_spotify_client_id
SPOTIFY_CLIENT_SECRET=your_spotify_client_secret
ANTHROPIC_API_KEY=your_anthropic_api_key
COHERE_API_KEY=your_cohere_api_key

# Optional — enables LangSmith tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT=spotify-rag-agent
```

| Key | Where to get it |
|---|---|
| `SPOTIFY_CLIENT_ID/SECRET` | [developer.spotify.com/dashboard](https://developer.spotify.com/dashboard) → Create App → Web API |
| `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com) |
| `COHERE_API_KEY` | [dashboard.cohere.com](https://dashboard.cohere.com) (free tier available) |
| `LANGCHAIN_API_KEY` | [smith.langchain.com](https://smith.langchain.com) → Settings → API Keys (optional) |

### 3. LangSmith tracing (optional)

If `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` are set, every pipeline run is traced automatically. The project is created on the first run — no manual setup needed in the UI.

Traced steps per run:
- `cohere_rank_results` — Cohere rerank scores for Spotify results
- `retrieve_and_rerank` — FAISS candidate fetch + Cohere cross-encoder rerank
- `classify_query_intent` — detected intent + extracted sub-questions
- All `ChatAnthropic` calls — full prompts, responses, token counts, latency

---

## Usage

### Spotify Content Explorer

```bash
# Single query
python main.py --query "mindfulness meditation"

# Interactive mode
python main.py

# With RAG evaluations
python main.py --query "mindfulness meditation" --eval
```

The output structure adapts to your query intent:

| Query type | Intent detected | Output shape |
|---|---|---|
| `"mindfulness meditation"` | discovery | Overview + topic sections + Where to Start |
| `"best sleep podcast for anxiety"` | recommendation | Direct pick first + justifications + Top 3 |
| `"podcast vs audiobook for stoicism"` | comparison | Verdict first + side-by-side + Our Verdict |
| `"everything about intermittent fasting"` | deep_dive | Synthesis + thematic sections + Key Takeaways |

---

### Interview Prep

```bash
# Resume PDF + JD PDF
python main.py --resume my_resume.pdf --jd job_description.pdf

# Resume PDF + JD as a text file
python main.py --resume my_resume.pdf --jd job_description.txt

# Resume PDF + raw JD text
python main.py --resume my_resume.pdf --jd "Senior ML Engineer, 5+ years Python..."

# Save report to file
python main.py --resume my_resume.pdf --jd job_description.txt --interview-output prep_report.txt

# With RAG evaluations
python main.py --resume my_resume.pdf --jd job_description.txt --eval
```

**The report contains 3 parts:**

**Part 1 — Targeted Interview Questions**
Identifies gaps between your resume and the JD, generates 2 focused questions per gap (behavioural + technical) with hints on what a strong answer covers. Grounded strictly in the provided documents.

**Part 2 — Suggested STAR Answers**
Drafts Situation → Task → Action → Result answers using only your actual resume experience. Does not fabricate projects, metrics, or skills not present in your resume.

**Part 3 — Preparation Strategy**
- Strengths to lead with, drawn from your resume
- Top gaps to study before the interview with specific action items
- Smart questions to ask the interviewer grounded in the JD
- Tactical tips specific to the role type

---

## Project Structure

```
spotify-rag-agent/
├── main.py               # CLI entry point
├── agent.py              # Spotify RAG pipeline orchestrator
├── interview_agent.py    # Interview prep pipeline orchestrator
├── query_intent.py       # Query intent classifier + intent-specific prompt templates
├── spotify_tools.py      # LangChain tools wrapping Spotify Web API
├── ranking.py            # Cohere rerank + composite scoring
├── embeddings_store.py   # FAISS vector store, Cohere embeddings, two-stage retrieval
├── resume_parser.py      # Resume + JD parsing (PDF, TXT, URL)
├── guardrails.py         # Input/output validation (no LLM cost)
├── rag_evals.py          # LLM-as-judge RAG evaluation metrics
├── requirements.txt      # Python dependencies
└── .env.example          # API key template
```

See [EVALS_AND_GUARDRAILS.md](EVALS_AND_GUARDRAILS.md) for a detailed guide to the evaluation and guardrails system.

---

## How the RAG Pipeline Works

### Spotify Mode

1. **Guardrails** — Validates query length, detects prompt injection
2. **Search** — Fetches top 10 podcasts + top 3 audiobooks from Spotify API
3. **Rank** — Cohere rerank scores each result for semantic relevance, combined with episode count and description richness in a weighted composite score
4. **Embed** — All results embedded with Cohere and indexed in FAISS
5. **Two-stage retrieval** — FAISS fetches a broad candidate pool (k×3), Cohere rerank re-scores and selects top-8 by cross-encoder relevance
6. **Intent classification** — Claude Haiku classifies the query as `discovery`, `recommendation`, `comparison`, or `deep_dive` and extracts sub-questions
7. **Intent-aware synthesis** — Claude Sonnet uses a prompt template shaped for the detected intent; sub-questions drive section headings rather than a fixed template
8. **Output guardrails** — Checks report length and detects LLM refusal phrases

### Interview Prep Mode

1. **Guardrails** — Validates resume file (PDF/TXT, non-empty) and JD input (file path, URL, or raw text ≥50 chars)
2. **Parse** — Extracts text from resume and JD
3. **Chunk** — Splits both into 400-char overlapping chunks tagged by source (`resume` / `jd`)
4. **Embed** — Chunks embedded with Cohere and indexed in FAISS
5. **Two-stage retrieval** — Per-pass: FAISS fetches k×3 candidates filtered by source, Cohere rerank selects top-10
6. **Synthesise** — Three Claude Sonnet calls, each grounded in strict no-fabrication instructions:
   - Gap analysis → targeted interview questions
   - STAR answers grounded in resume evidence only
   - Preparation strategy tied to JD requirements
7. **Output guardrails** — Checks all three report sections are present
