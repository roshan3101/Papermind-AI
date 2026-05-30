import fitz  # PyMuPDF
import re
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ParsedPage:
    page_number: int
    text: str
    char_count: int


@dataclass
class ParsedDocument:
    title: str
    authors: Optional[str]
    abstract: Optional[str]
    year: Optional[int]
    pages: list[ParsedPage]
    full_text: str
    page_count: int
    metadata: dict


def extract_year(text: str) -> Optional[int]:
    matches = re.findall(r"\b(19\d{2}|20[0-2]\d)\b", text[:2000])
    return int(matches[0]) if matches else None


def extract_abstract(text: str) -> Optional[str]:
    pattern = r"(?i)abstract[.\s]*\n(.*?)(?=\n(?:1\.?\s*introduction|keywords|1\s+introduction))"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()[:2000]
    return None


def extract_title_authors(doc: fitz.Document, first_page_text: str) -> tuple[str, Optional[str]]:
    # Try PDF metadata first
    meta = doc.metadata
    title = meta.get("title", "").strip()
    author = meta.get("author", "").strip()

    if not title:
        # Use first non-empty line as title
        lines = [l.strip() for l in first_page_text.split("\n") if l.strip()]
        title = lines[0] if lines else "Untitled"

    if not author:
        # Heuristic: second line often has authors
        lines = [l.strip() for l in first_page_text.split("\n") if l.strip()]
        author = lines[1] if len(lines) > 1 else None

    return title[:500], author[:500] if author else None


def parse_pdf(file_path: str) -> ParsedDocument:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    doc = fitz.open(str(path))
    pages: list[ParsedPage] = []
    full_text_parts = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        pages.append(ParsedPage(
            page_number=page_num + 1,
            text=text,
            char_count=len(text),
        ))
        full_text_parts.append(text)

    full_text = "\n\n".join(full_text_parts)
    first_page_text = pages[0].text if pages else ""

    title, authors = extract_title_authors(doc, first_page_text)
    abstract = extract_abstract(full_text)
    year = extract_year(first_page_text)

    meta = doc.metadata or {}
    doc.close()

    logger.info("pdf_parsed", file=path.name, pages=len(pages), chars=len(full_text))

    return ParsedDocument(
        title=title,
        authors=authors,
        abstract=abstract,
        year=year,
        pages=pages,
        full_text=full_text,
        page_count=len(pages),
        metadata={
            "subject": meta.get("subject", ""),
            "creator": meta.get("creator", ""),
            "keywords": meta.get("keywords", ""),
        },
    )
