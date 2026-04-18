"""
interview_agent.py
Orchestrates the interview prep pipeline:
  1. Input guardrails (resume file validation, JD input validation)
  2. Parse resume + job description (PDF, TXT, or URL)
  3. Chunk into 400-char overlapping segments tagged by source (resume / jd)
  4. Cohere embeddings + FAISS vector store
  5. Two-stage retrieval per pass — FAISS candidate pool → Cohere rerank,
     filtered by source (resume or jd chunks), k=10
  6. Three Claude synthesis calls:
       - Gap analysis → targeted interview questions
       - STAR answers grounded in resume evidence only
       - Preparation strategy tied to JD requirements
  7. Output guardrails (all three report sections present, length check)
"""

import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

from resume_parser import parse_resume, parse_jd, build_interview_documents
from embeddings_store import build_vector_store_from_docs, retrieve_relevant_context, retrieve_and_rerank
from guardrails import validate_resume_file, validate_jd_input, check_output_completeness

load_dotenv()

# ── Prompt templates ────────────────────────────────────────────────────────────

_QUESTIONS_PROMPT = """You are an expert technical interviewer.

You have been given excerpts from a candidate's resume and the job description they are applying for.

STRICT GROUNDING RULES:
- Identify gaps only from what is explicitly stated or clearly absent in the excerpts below.
- Do NOT assume skills, tools, or experience the candidate has or lacks beyond what the excerpts show.
- Do NOT reference technologies, responsibilities, or requirements that do not appear in the JD excerpts.
- If the excerpts are insufficient to identify 5 clear gaps, identify as many as the excerpts support.

Your task:
1. Identify the most significant gaps between what the JD excerpts require and what the resume excerpts demonstrate.
2. For each gap, write 2 targeted interview questions a hiring manager would ask to probe that gap.
   Mix behavioural ("Tell me about a time...") and technical questions.
3. For each question, add a one-line hint: what a strong answer should cover, based on the JD excerpts.

Format strictly as:

GAP 1: <gap description>
  Q1: <question>
      Hint: <what a strong answer should cover>
  Q2: <question>
      Hint: <what a strong answer should cover>

GAP 2: ...

═══════════════════════════
RESUME EXCERPTS:
{resume_context}

═══════════════════════════
JOB DESCRIPTION EXCERPTS:
{jd_context}
═══════════════════════════

Generate the interview questions now:"""


_ANSWERS_PROMPT = """You are a career coach helping a candidate prepare for an interview.

Using the STAR method (Situation → Task → Action → Result), draft a suggested answer for each
interview question below.

STRICT GROUNDING RULES — you must follow these without exception:
- Every Situation, Task, Action, and Result you write must come directly from the resume excerpts below.
- Do NOT invent project names, company names, team sizes, timelines, percentages, or metrics
  unless they appear verbatim in the resume excerpts.
- Do NOT attribute tools, languages, or frameworks to the candidate unless explicitly listed in the excerpts.
- If the resume excerpts do not contain direct evidence for a question, write:
  "Resume evidence is limited here. Frame your answer around: [quote the closest relevant excerpt]."
  Do NOT fill the gap with fabricated experience.

INTERVIEW QUESTIONS:
{questions}

═══════════════════════════
CANDIDATE'S RELEVANT RESUME EXCERPTS:
{resume_context}
═══════════════════════════

Write a suggested STAR answer for each question, labelled Q1, Q2, ... in the same order:"""


_STRATEGY_PROMPT = """You are a senior career coach reviewing a candidate's profile against a target role.

STRICT GROUNDING RULES:
- Strengths must be drawn only from what is explicitly stated in the resume excerpts.
- Gaps must be derived only from requirements explicitly stated in the JD excerpts that are
  absent or underrepresented in the resume excerpts.
- Do NOT reference technologies, companies, or responsibilities not present in the excerpts.
- Smart questions to ask the interviewer must be grounded in the JD excerpts (role, team, expectations).
- Tactical tips must be specific to the role type described in the JD excerpts, not generic advice.

Provide a concise, actionable preparation plan:

1. STRENGTHS TO HIGHLIGHT
   - What the candidate should lead with, drawn directly from the resume excerpts (3-4 bullets).

2. GAPS TO ADDRESS BEFORE THE INTERVIEW
   - Top 3 areas where the JD excerpts signal a requirement the resume excerpts do not cover,
     each with a specific study or practice action item.

3. SMART QUESTIONS TO ASK THE INTERVIEWER
   - 3 questions grounded in what the JD excerpts reveal about the role and team.

4. MINDSET & TACTICAL TIPS
   - 2-3 practical tips specific to the role type described in the JD excerpts.

═══════════════════════════
CANDIDATE PROFILE (resume excerpts):
{resume_context}

TARGET ROLE (JD excerpts):
{jd_context}
═══════════════════════════

Write the preparation plan now:"""


class InterviewPrepAgent:
    """
    End-to-end interview preparation pipeline.
    Parses resume + JD → embeds into FAISS → generates questions, answers, strategy via Claude.
    """

    def __init__(self) -> None:
        self.llm = ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
            anthropic_api_key=os.environ["ANTHROPIC_API_KEY"],
            temperature=0.4,
            max_tokens=4096,
        )

    def run(self, resume_path: str, jd: str, run_evals: bool = False) -> str:
        """
        Execute the full interview prep pipeline.

        Args:
            resume_path: Path to a PDF or TXT resume file.
            jd:          Path to a JD text file, or the raw JD string itself.
            run_evals:   If True, run LLM-as-judge RAG evaluations and print
                         the eval report after the pipeline completes.

        Returns:
            A comprehensive markdown-formatted interview prep report.
        """
        print("\n" + "=" * 60)
        print("  Interview Prep Pipeline")
        print("=" * 60 + "\n")

        # ── Input guardrails ────────────────────────────────────────────────────
        resume_guard = validate_resume_file(resume_path)
        resume_guard.print_warnings()
        resume_guard.raise_if_failed()

        jd_guard = validate_jd_input(jd)
        jd_guard.print_warnings()
        jd_guard.raise_if_failed()

        # ── Step 1: Parse ───────────────────────────────────────────────────────
        print("► Step 1: Parsing resume and job description...")
        resume_text = parse_resume(resume_path)
        jd_text = parse_jd(jd)
        print(f"  Resume: {len(resume_text)} chars | JD: {len(jd_text)} chars")

        # ── Step 2: Chunk + embed into FAISS ───────────────────────────────────
        print("► Step 2: Chunking and embedding with Cohere into FAISS...")
        docs = build_interview_documents(resume_text, jd_text)
        resume_docs = [d for d in docs if d.metadata["source"] == "resume"]
        jd_docs = [d for d in docs if d.metadata["source"] == "jd"]
        print(f"  {len(resume_docs)} resume chunks + {len(jd_docs)} JD chunks = {len(docs)} total")
        vector_store = build_vector_store_from_docs(docs)

        # ── Step 3: Generate interview questions ───────────────────────────────
        print("► Step 3: Generating targeted interview questions...")
        resume_ctx = self._retrieve_by_source(vector_store, "skills experience background", source="resume")
        jd_ctx = self._retrieve_by_source(vector_store, "requirements responsibilities qualifications", source="jd")

        questions_raw: str = self.llm.invoke(
            _QUESTIONS_PROMPT.format(resume_context=resume_ctx, jd_context=jd_ctx)
        ).content

        # ── Step 4: Generate suggested answers ─────────────────────────────────
        print("► Step 4: Generating STAR answers grounded in resume...")
        answer_resume_ctx = self._retrieve_by_source(
            vector_store, "achievements projects experience skills", source="resume", k=8
        )
        answers_raw: str = self.llm.invoke(
            _ANSWERS_PROMPT.format(questions=questions_raw, resume_context=answer_resume_ctx)
        ).content

        # ── Step 5: Generate preparation strategy ──────────────────────────────
        print("► Step 5: Generating personalised preparation strategy...")
        strategy_resume_ctx = self._retrieve_by_source(vector_store, "candidate strengths profile", source="resume")
        strategy_jd_ctx = self._retrieve_by_source(vector_store, "role requirements must have skills", source="jd")

        strategy_raw: str = self.llm.invoke(
            _STRATEGY_PROMPT.format(resume_context=strategy_resume_ctx, jd_context=strategy_jd_ctx)
        ).content

        # Output guardrail
        report = self._assemble_report(questions_raw, answers_raw, strategy_raw)
        output_guard = check_output_completeness(
            report,
            expected_sections=["PART 1", "PART 2", "PART 3"],
        )
        output_guard.print_warnings()
        output_guard.raise_if_failed()

        print("\n► Done!\n")

        # ── Optional: RAG evaluations ───────────────────────────────────────────
        if run_evals:
            from rag_evals import RAGEvaluator
            # Use reranked scores (already 0-1) for retrieval precision
            _, reranked_docs_with_scores = retrieve_and_rerank(
                vector_store, "skills experience requirements", k=10
            )
            evaluator = RAGEvaluator()
            eval_results = evaluator.run_interview_evals(
                retrieved_docs=reranked_docs_with_scores,
                context=resume_ctx + "\n\n" + jd_ctx,
                resume_context=answer_resume_ctx,
                star_answers=answers_raw,
                scores_are_similarities=True,
            )
            print(evaluator.format_report(eval_results))

        return report

    def _retrieve_by_source(self, vector_store, query: str, source: str, k: int = 10) -> str:
        """
        Retrieve top-k chunks filtered to a specific source (resume or jd).

        Uses two-stage retrieval: FAISS fetches a broad candidate pool (k×3),
        then Cohere rerank re-scores and selects the top-k most relevant chunks.
        This replaces the old FAISS L2 distance scoring with cross-encoder accuracy.
        """
        context, _ = retrieve_and_rerank(
            vector_store, query, k=k, source_filter=source
        )
        return context

    def _assemble_report(self, questions: str, answers: str, strategy: str) -> str:
        """Combine the three LLM outputs into a single formatted markdown report."""
        divider = "\n" + "═" * 60 + "\n"
        return (
            f"# Interview Preparation Report\n"
            f"{divider}"
            f"## PART 1 — Targeted Interview Questions\n\n{questions}"
            f"{divider}"
            f"## PART 2 — Suggested STAR Answers\n\n{answers}"
            f"{divider}"
            f"## PART 3 — Preparation Strategy\n\n{strategy}\n"
        )
