# Spotify RAG Agent

A RAG (Retrieval-Augmented Generation) agent that combines the **Spotify API**, **Cohere**, and **Claude** to deliver two features:

1. **Spotify Content Explorer** — search any topic and get a ranked, synthesised report of the top podcasts and audiobooks on Spotify.
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
          │  1. Spotify API Search       │  │  1. Parse Resume + JD    │
          │     (spotify_tools.py)       │  │     (resume_parser.py)   │
          ├──────────────────────────────┤  ├──────────────────────────┤
          │  2. Cohere Rerank            │  │  2. Chunk + Tag by source│
          │     (ranking.py)             │  │     (resume_parser.py)   │
          ├──────────────────────────────┤  ├──────────────────────────┤
          │  3. Cohere Embeddings        │  │  3. Cohere Embeddings    │
          │  +  FAISS Vector Store       │  │  +  FAISS Vector Store   │
          │     (embeddings_store.py)    │  │     (embeddings_store.py)│
          ├──────────────────────────────┤  ├──────────────────────────┤
          │  4. Semantic Retrieval       │  │  4. Source-filtered      │
          │     (top-k chunks)           │  │     Retrieval per pass   │
          ├──────────────────────────────┤  ├──────────────────────────┤
          │  5. Claude Synthesis         │  │  5. Claude: Questions    │
          │     (comprehensive report)   │  │         + STAR Answers   │
          └──────────────────────────────┘  │         + Prep Strategy  │
                                            └──────────────────────────┘
```

---

## Tech Stack

| Component | Tool |
|---|---|
| LLM | Claude (Anthropic) via `langchain-anthropic` |
| Embeddings | Cohere `embed-english-v3.0` |
| Reranking | Cohere `rerank-english-v3.0` |
| Vector Store | FAISS (in-memory) |
| Spotify Search | Spotipy (Spotify Web API) |
| PDF Parsing | pypdf |
| Web Scraping | LangChain `WebBaseLoader` + BeautifulSoup |
| Orchestration | LangChain |

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

Copy the example env file and fill in your keys:

```bash
cp .env.example .env
```

Edit `.env`:

```env
SPOTIFY_CLIENT_ID=your_spotify_client_id
SPOTIFY_CLIENT_SECRET=your_spotify_client_secret
ANTHROPIC_API_KEY=your_anthropic_api_key
COHERE_API_KEY=your_cohere_api_key
```

| Key | Where to get it |
|---|---|
| `SPOTIFY_CLIENT_ID/SECRET` | [developer.spotify.com/dashboard](https://developer.spotify.com/dashboard) → Create App → Web API |
| `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com) |
| `COHERE_API_KEY` | [dashboard.cohere.com](https://dashboard.cohere.com) (free tier available) |

---

## Usage

### Spotify Content Explorer

Search any topic and get a ranked report of the top 10 podcasts and top 3 audiobooks:

```bash
# Single query
python main.py --query "mindfulness meditation"

# Interactive mode (keeps prompting for topics)
python main.py
```

**Example output sections:**
- Overview of the topic on Spotify
- Top Podcasts — what makes each unique, who it's for
- Top Audiobooks — authors, themes, key takeaways
- Key Themes across all content
- Recommendations — top picks by depth and relevance

---

### Interview Prep

Upload your resume and a job description to get a personalised prep report:

```bash
# Resume PDF + JD PDF
python main.py --resume my_resume.pdf --jd job_description.pdf

# Resume PDF + JD as a text file
python main.py --resume my_resume.pdf --jd job_description.txt

# Resume PDF + raw JD text
python main.py --resume my_resume.pdf --jd "Senior ML Engineer, 5+ years Python, Spark..."

# Save the report to a file
python main.py --resume my_resume.pdf --jd job_description.pdf --interview-output prep_report.txt
```

**The report contains 3 parts:**

**Part 1 — Targeted Interview Questions**
Identifies the top 5 gaps between your resume and the JD, then generates 2 focused questions per gap (behavioural + technical) with hints on what a strong answer covers.

**Part 2 — Suggested STAR Answers**
For each question, drafts a Situation → Task → Action → Result answer grounded in your actual resume experience. Does not fabricate projects or metrics.

**Part 3 — Preparation Strategy**
- Strengths to lead with in the interview
- Top 3 gaps to study before the interview, with specific action items
- Smart questions to ask the interviewer
- Tactical tips for this role type

---

## Project Structure

```
spotify-rag-agent/
├── main.py               # CLI entry point
├── agent.py              # Spotify RAG pipeline orchestrator
├── spotify_tools.py      # LangChain tools wrapping Spotify Web API
├── ranking.py            # Cohere rerank + composite scoring
├── embeddings_store.py   # FAISS vector store with Cohere embeddings
├── interview_agent.py    # Interview prep pipeline orchestrator
├── resume_parser.py      # Resume + JD parsing (PDF, TXT, URL)
├── requirements.txt      # Python dependencies
└── .env.example          # API key template
```

---

## How the RAG Pipeline Works

### Spotify Mode
1. **Search** — Fetches top 10 podcasts + top 3 audiobooks from Spotify API
2. **Rank** — Cohere rerank scores each result for semantic relevance, combined with episode count (popularity) and description length (richness) in a weighted composite score
3. **Embed** — All results are embedded with Cohere and indexed in an in-memory FAISS store
4. **Retrieve** — Top-6 most semantically similar chunks are retrieved for the user's query
5. **Synthesise** — Claude writes a comprehensive, structured report from the ranked results and retrieved context

### Interview Prep Mode
1. **Parse** — Extracts text from resume and JD (PDF, TXT, or URL)
2. **Chunk** — Splits both into 400-char overlapping chunks tagged by source (`resume` / `jd`)
3. **Embed** — Chunks are embedded with Cohere and indexed in FAISS
4. **Retrieve** — Three separate retrieval passes with source-filtered queries (resume chunks vs JD chunks)
5. **Synthesise** — Three Claude calls: gap analysis + questions → STAR answers → prep strategy
