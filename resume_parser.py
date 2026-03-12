"""
resume_parser.py
Parses resume (PDF or TXT) and job description (TXT file or raw string) into
LangChain Documents ready for FAISS embedding.
"""

import os
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader


def parse_resume(path: str) -> str:
    """Extract text from a PDF or TXT resume file."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(path)
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n".join(pages).strip()
        if not text:
            raise ValueError(
                f"Could not extract text from '{path}'. "
                "Make sure it is a text-based PDF (not scanned)."
            )
        return text
    else:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()


def parse_jd(path_or_url_or_text: str) -> str:
    """
    Accept a job description from three sources (auto-detected):
      1. URL  – fetches the page and extracts visible text via BeautifulSoup.
      2. File path – reads a local .txt file.
      3. Raw string – uses the value directly as JD content.

    Args:
        path_or_url_or_text: A URL (http/https), a file path, or the raw JD text.

    Returns:
        The extracted JD text as a plain string.
    """
    # ── URL ────────────────────────────────────────────────────────────────────
    if path_or_url_or_text.startswith("http://") or path_or_url_or_text.startswith("https://"):
        print(f"  Fetching JD from URL: {path_or_url_or_text}")
        loader = WebBaseLoader(web_paths=[path_or_url_or_text])
        docs = loader.load()
        if not docs:
            raise ValueError(f"Could not fetch content from URL: {path_or_url_or_text}")
        text = "\n".join(d.page_content for d in docs).strip()
        if not text:
            raise ValueError(f"Fetched page at '{path_or_url_or_text}' is empty or unreadable.")
        return text

    # ── File path ──────────────────────────────────────────────────────────────
    if os.path.isfile(path_or_url_or_text):
        ext = os.path.splitext(path_or_url_or_text)[1].lower()
        if ext == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(path_or_url_or_text)
            text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
            if not text:
                raise ValueError(
                    f"Could not extract text from '{path_or_url_or_text}'. "
                    "Make sure it is a text-based PDF (not scanned)."
                )
            return text
        with open(path_or_url_or_text, "r", encoding="utf-8") as f:
            return f.read().strip()

    # ── Raw text ───────────────────────────────────────────────────────────────
    return path_or_url_or_text.strip()


def build_interview_documents(
    resume_text: str,
    jd_text: str,
    chunk_size: int = 400,
    chunk_overlap: int = 80,
) -> list[Document]:
    """
    Chunk resume and JD text into overlapping segments and tag each chunk
    with its source so the agent can filter during retrieval.

    Args:
        resume_text:  Full text of the candidate's resume.
        jd_text:      Full text of the job description.
        chunk_size:   Max characters per chunk (default 400 ≈ 2-3 bullet points).
        chunk_overlap: Overlap between consecutive chunks (default 80).

    Returns:
        A list of Documents with metadata {"source": "resume"|"jd", "chunk_index": int}.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    docs: list[Document] = []

    for i, chunk in enumerate(splitter.split_text(resume_text)):
        docs.append(Document(page_content=chunk, metadata={"source": "resume", "chunk_index": i}))

    for i, chunk in enumerate(splitter.split_text(jd_text)):
        docs.append(Document(page_content=chunk, metadata={"source": "jd", "chunk_index": i}))

    return docs
