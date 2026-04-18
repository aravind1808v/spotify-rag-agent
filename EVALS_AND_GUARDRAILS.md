# Evals and Guardrails

This document covers the evaluation and guardrails system added to the Spotify RAG Agent.
The two systems serve different purposes:

- **Guardrails** (`guardrails.py`) — fast, heuristic checks that run on every pipeline execution at no extra API cost. They catch bad inputs before they waste API calls and verify outputs are substantive before returning them to the user.
- **Evaluations** (`rag_evals.py`) — LLM-as-judge metrics that score the quality of the RAG pipeline. They run on demand via `--eval` and use Claude Haiku as the judge to keep costs low.

---

## Guardrails

### How to run

Guardrails run automatically on every pipeline call — no flag needed. Errors abort the pipeline with a clear message. Warnings are printed but do not stop execution.

```
[WARN] Single-word query may return broad results. Consider adding context.
```

```
ValueError: Guardrail violations:
  • Resume file not found: 'my_cv.pdf'
```

### Input guardrails

#### `validate_query(query)` — Spotify mode

| Check | Behaviour |
|---|---|
| Empty string | Hard error |
| < 3 characters | Hard error |
| > 500 characters | Hard error |
| Prompt injection patterns | Hard error (`ignore previous instructions`, `you are now a`, etc.) |
| Single word | Soft warning |

#### `validate_resume_file(path)` — Interview mode

| Check | Behaviour |
|---|---|
| Empty path | Hard error |
| File not found | Hard error |
| Unsupported extension (not `.pdf` / `.txt`) | Hard error |
| Empty file (0 bytes) | Hard error |
| File > 10 MB | Soft warning |

#### `validate_jd_input(jd)` — Interview mode

Accepts three input types, auto-detected:

| Input type | Detection | Behaviour |
|---|---|---|
| URL (`http://` / `https://`) | Prefix match | Passed through; fetched at parse time |
| File path (`.pdf`, `.txt`, `.md`, `.docx`) | Extension match | Checked for existence; hard error if not found |
| Raw text | Fallback | Hard error if < 50 characters |

The extension-before-existence check prevents a common mistake: passing `jd.pdf` when the file doesn't exist at the current path produces "file not found" rather than the misleading "text too short" error.

### Retrieval quality guardrail

#### `check_retrieval_quality(scores, scores_are_cohere)`

Runs after retrieval. Warns if the retrieved chunk set is low quality.

**FAISS similarity scores (0–100 pct):** absolute threshold check.

| Condition | Behaviour |
|---|---|
| > 50% of chunks below 30% similarity | Soft warning with average score |

**Cohere rerank scores (0–1):** distribution-aware check. Cohere scores are relative — an absolute threshold is meaningless (a score of 0.08 can be the most relevant document). Two conditions trigger a warning:

| Condition | Behaviour |
|---|---|
| Top score < 0.05 | Soft warning — nothing in the corpus matched the query |
| > 50% of chunks below 30% of the top score | Soft warning — major drop-off in quality across the retrieved set |

Pass `scores_are_cohere=True` when scores come from `retrieve_and_rerank`.

### Output guardrails

#### `check_output_completeness(output, min_length, expected_sections)`

Runs after each LLM synthesis call.

| Check | Behaviour |
|---|---|
| Empty output | Hard error |
| Output < 100 characters | Hard error |
| Expected section heading missing | Soft warning |
| LLM refusal phrase detected (`i cannot`, `as an ai`, etc.) | Soft warning |

---

## RAG Evaluations

### How to run

Add `--eval` to any pipeline command:

```bash
# Spotify mode
python main.py --query "machine learning" --eval

# Interview mode
python main.py --resume resume.pdf --jd jd.txt --eval
```

The eval report is printed to stdout after the pipeline output:

```
╔══════════════════════════════════════════════════════════════════════╗
║  RAG Evaluation Report                                               ║
╠══════════════════════════════════════════════════════════════════════╣
║  Metric                    Score  Label  Threshold                  ║
║  ────────────────────────────────────────────────────────────────── ║
║  context_relevance         0.810  ✓ PASS       0.50                 ║
║  faithfulness              1.000  ✓ PASS       0.70                 ║
║  answer_relevance          0.720  ✓ PASS       0.70                 ║
║  retrieval_precision       0.780  ✓ PASS       0.50                 ║
╚══════════════════════════════════════════════════════════════════════╝

  Reasoning
  ──────────────────────────────────────────────────────────────────────
  ✓ context_relevance:
      Retrieved passages are on-topic for the query; descriptions
      clearly match the subject area.
  ...
```

### Judge model

All LLM-based evaluations use **Claude Haiku** (`claude-haiku-4-5`) as the judge — not the same model used for synthesis. This keeps evaluation costs low and avoids the judge being biased towards the synthesis model's output style.

---

### Metric reference

#### 1. Context Relevance

**What it measures:** Are the retrieved chunks actually relevant to the user's query?

**How it works:** The judge reads each retrieved passage (up to 300 chars each) and rates it 0–1 against the query using a rubric calibrated for Spotify metadata. Passages are descriptive catalogue entries (titles, publishers, descriptions) — the rubric accounts for this format rather than expecting dense explanatory text.

**Scoring rubric:**
| Score | Meaning |
|---|---|
| 1.0 | Passage is about a show directly on this topic; description clearly matches the query |
| 0.7 | Passage is about a related show; partial match |
| 0.4 | Topic mentioned incidentally or loosely related |
| 0.0 | Completely different topic |

**Threshold:** 0.50
**Mode:** Spotify only

---

#### 2. Faithfulness

**What it measures:** Is the generated answer grounded in the retrieved context? Flags hallucinations.

**How it works:** The judge extracts up to 10 key factual claims from the answer, then marks each as `GROUNDED` (supported by context) or `NOT_GROUNDED`. The score is `grounded / total`.

**What counts as not grounded:**
- Statistics, episode counts, or metrics not in the passages
- Author names or publisher details not explicitly stated
- Audience descriptions or themes inferred rather than quoted

Any `not_grounded_examples` are printed below the eval table for review.

**Threshold:** 0.70
**Mode:** Spotify only

---

#### 3. Answer Relevance

**What it measures:** Does the answer actually address what the user asked?

**How it works:** The judge reads the first 4000 characters of the generated answer against the original query. The rubric is calibrated to recognise that well-structured topic reports are valid direct answers for exploration queries (not just conversational responses).

**Scoring rubric:**
| Score | Meaning |
|---|---|
| 1.0 | Opens by directly answering the query; every section ties back to it |
| 0.8 | Clearly addresses the query; minor sections feel generic |
| 0.6 | Covers the topic but structured generically rather than as a direct answer |
| 0.4 | Mentions the topic but reads more like a catalogue |
| 0.2 | Tangentially related; query intent not addressed |
| 0.0 | Completely irrelevant |

**Threshold:** 0.70
**Mode:** Spotify only

---

#### 4. Retrieval Precision

**What it measures:** How good is the quality of the retrieved chunk set overall?

**How it works:** Differs by retrieval method:

**For FAISS L2 scores** (absolute threshold):
Converts L2 distances to similarity percentages and checks what fraction exceed 30%.

**For Cohere rerank scores** (distribution-aware):
Cohere scores are relative, not absolute — a score of 0.08 can still be the most relevant document. An absolute threshold would be meaningless. Instead two signals are combined:

- *Score concentration (60% weight)*: fraction of total relevance mass sitting in the top half of retrieved docs. A tight cluster at the top means the reranker found genuinely better chunks.
- *Mean-relative precision (40% weight)*: fraction of chunks scoring above 50% of the mean. Identifies chunks meaningfully better than the average retrieval.

**Threshold:** 0.50
**Mode:** Spotify + Interview

---

#### 5. Resume Grounding

**What it measures:** Are the STAR answers grounded in the candidate's actual resume? Flags fabricated claims.

**How it works:** The judge identifies all concrete claims in the STAR answers (projects, metrics, tools, achievements) and checks each against the resume excerpts. Score is `grounded / total`.

**What counts as fabricated:**
- Project names not in the resume
- Specific metrics or percentages invented
- Tools or frameworks attributed to the candidate but not listed
- Job titles or team sizes not mentioned

Any `fabricated_examples` are printed below the eval table.

**Threshold:** 0.75
**Mode:** Interview only

---

### Metric coverage by mode

| Metric | Spotify | Interview |
|---|---|---|
| context_relevance | ✓ | — |
| faithfulness | ✓ | — |
| answer_relevance | ✓ | — |
| retrieval_precision | ✓ | ✓ |
| resume_grounding | — | ✓ |

---

## Design decisions

**Why separate guardrails from evals?**

Guardrails are always-on safety checks — they should be fast, free, and never slow down a normal run. Evaluations use LLM judge calls which add latency and API cost. Keeping them separate means guardrails can block bad inputs immediately while evals remain opt-in for when you want to measure quality.

**Why Claude Haiku as the judge?**

Haiku is significantly cheaper and faster than Sonnet. For structured JSON scoring tasks (rate this 0-1, list these claims) it performs comparably to larger models. Using a smaller judge also reduces the risk of the judge simply agreeing with the synthesis model's output style.

**Why distribution-aware retrieval precision for Cohere?**

Cohere rerank scores are cross-encoder outputs trained to rank documents relative to each other for a query. They are not calibrated probabilities — a score of 0.05 does not mean "5% similar." Applying an absolute threshold (e.g. must be > 0.30) would almost always fail for Cohere, even when retrieval is excellent. The distribution-aware approach measures the *shape* of the score distribution instead.

**Why 4000 chars for the answer relevance judge?**

The first 1500 chars of a structured report is almost always the introduction/overview — the most generic part. The Recommendations section, which is the most query-relevant, typically appears later. Increasing to 4000 chars ensures the judge sees the full answer before scoring.
