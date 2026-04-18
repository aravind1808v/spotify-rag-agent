"""
main.py
CLI entry point for the Spotify RAG chatbot.

Usage:
    python main.py
    python main.py --query "mindfulness meditation"
"""

import argparse
import logging
import os
import sys

# Load .env before any LangSmith/LangChain imports so LANGCHAIN_API_KEY
# is present when the tracing client initialises at import time.
from dotenv import load_dotenv
load_dotenv()

# Suppress noisy info messages from sagemaker and langchain WebBaseLoader
logging.getLogger("sagemaker").setLevel(logging.WARNING)
logging.getLogger("sagemaker.config").setLevel(logging.WARNING)
os.environ.setdefault("USER_AGENT", "SpotifyRAGAgent/1.0")

from agent import SpotifyRAGAgent


def interactive_loop(agent: SpotifyRAGAgent, run_evals: bool = False) -> None:
    """Run a continuous interactive Q&A session."""
    print("\n" + "=" * 60)
    print("  Spotify RAG Chatbot")
    print("  Powered by Claude + Cohere + Spotify")
    print("  Type 'quit' or 'exit' to stop.")
    if run_evals:
        print("  [eval mode ON — RAG evaluations will run after each query]")
    print("=" * 60 + "\n")

    while True:
        try:
            query = input("Enter a topic or question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        report = agent.run(query, run_evals=run_evals)
        print("\n" + "=" * 60)
        print(report)
        print("=" * 60 + "\n")


def main() -> None:
    """Parse CLI arguments and dispatch to Spotify RAG mode or interview prep mode."""
    parser = argparse.ArgumentParser(
        description="Spotify RAG chatbot – top podcasts & audiobooks on any topic. "
                    "Also supports interview prep from a resume + job description."
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Run a single Spotify query and exit (skip interactive mode).",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to your resume file (PDF or TXT) for interview prep mode.",
    )
    parser.add_argument(
        "--jd",
        type=str,
        default=None,
        help="Path to a job description TXT file, or paste the JD as a quoted string.",
    )
    parser.add_argument(
        "--interview-output",
        type=str,
        default=None,
        help="Optional path to save the interview prep report (e.g. prep_report.txt).",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        default=False,
        help=(
            "Run LLM-as-judge RAG evaluations after the pipeline completes. "
            "Prints a scored report covering context relevance, faithfulness, "
            "answer relevance, retrieval precision, and (interview mode) resume grounding. "
            "Uses claude-haiku for the judge calls."
        ),
    )
    args = parser.parse_args()

    # ── Interview prep mode ─────────────────────────────────────────────────────
    if args.resume or args.jd:
        if not args.resume or not args.jd:
            parser.error("Both --resume and --jd are required for interview prep mode.")
        from interview_agent import InterviewPrepAgent
        agent = InterviewPrepAgent()
        report = agent.run(resume_path=args.resume, jd=args.jd, run_evals=args.eval)
        print("\n" + "=" * 60)
        print(report)
        print("=" * 60)
        if args.interview_output:
            with open(args.interview_output, "w", encoding="utf-8") as f:
                f.write(report)
            print(f"\nReport saved to: {args.interview_output}")
        sys.exit(0)

    # ── Spotify RAG mode ────────────────────────────────────────────────────────
    agent = SpotifyRAGAgent()

    if args.query:
        report = agent.run(args.query, run_evals=args.eval)
        print("\n" + "=" * 60)
        print(report)
        print("=" * 60)
        sys.exit(0)
    else:
        interactive_loop(agent, run_evals=args.eval)


if __name__ == "__main__":
    main()
