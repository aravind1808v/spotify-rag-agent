"""
interview_agent.py
Orchestrates the interview prep pipeline:
  1. Parse resume + job description
  2. Chunk and embed into FAISS
  3. Generate targeted interview questions (gap analysis)
  4. Generate suggested answers grounded in the candidate's resume
  5. Generate a personalised preparation strategy
"""

import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

from resume_parser import parse_resume, parse_jd, build_interview_documents
from embeddings_store import build_vector_store_from_docs, retrieve_relevant_context

load_dotenv()

# ── Prompt templates ────────────────────────────────────────────────────────────

_QUESTIONS_PROMPT = """You are an expert technical interviewer.

You have been given excerpts from a candidate's resume and the job description they are applying for.

Your task:
1. Identify the 5 most significant gaps between what the JD requires and what the resume demonstrates.
2. For each gap, write 2 targeted interview questions a hiring manager would ask to probe that gap.
   Mix behavioural ("Tell me about a time...") and technical questions.
3. For each question, add a one-line hint: what a strong answer should cover.

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
interview question below. Ground every answer in the candidate's actual resume experience.
Do NOT invent projects, metrics, or skills not mentioned in the resume excerpts.
If the resume is thin on direct evidence for a question, suggest how to frame adjacent experience.

INTERVIEW QUESTIONS:
{questions}

═══════════════════════════
CANDIDATE'S RELEVANT RESUME EXCERPTS:
{resume_context}
═══════════════════════════

Write a suggested STAR answer for each question, labelled Q1, Q2, ... in the same order:"""


_STRATEGY_PROMPT = """You are a senior career coach reviewing a candidate's profile against a target role.

Provide a concise, actionable preparation plan:

1. STRENGTHS TO HIGHLIGHT
   - What the candidate should lead with in the interview (3-4 bullets drawn from resume strengths).

2. GAPS TO ADDRESS BEFORE THE INTERVIEW
   - Top 3 areas to study or practise, each with a specific action item
     (e.g. "Practice system design for distributed caching using ByteByteGo chapter 6").

3. SMART QUESTIONS TO ASK THE INTERVIEWER
   - 3 questions that show genuine research into the role and company.

4. MINDSET & TACTICAL TIPS
   - 2-3 practical tips tailored to this specific role type.

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

    def run(self, resume_path: str, jd: str) -> str:
        """
        Execute the full interview prep pipeline.

        Args:
            resume_path: Path to a PDF or TXT resume file.
            jd:          Path to a JD text file, or the raw JD string itself.

        Returns:
            A comprehensive markdown-formatted interview prep report.
        """
        print("\n" + "=" * 60)
        print("  Interview Prep Pipeline")
        print("=" * 60 + "\n")

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

        print("\n► Done!\n")
        return self._assemble_report(questions_raw, answers_raw, strategy_raw)

    def _retrieve_by_source(self, vector_store, query: str, source: str, k: int = 6) -> str:
        """Retrieve top-k chunks filtered to a specific source (resume or jd)."""
        results = vector_store.similarity_search_with_score(query, k=k * 2)
        filtered = [
            (doc, score) for doc, score in results if doc.metadata.get("source") == source
        ][:k]
        parts = []
        for doc, score in filtered:
            similarity_pct = max(0.0, 1.0 - score / 2.0) * 100
            parts.append(f"[{similarity_pct:.0f}%] {doc.page_content}")
        return "\n\n".join(parts) if parts else f"No {source} content retrieved."

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
