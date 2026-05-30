from dataclasses import dataclass
from typing import Optional
import re

from app.core.config import settings
from app.core.logging import get_logger
from app.rag.pdf_parser import ParsedDocument, ParsedPage

logger = get_logger(__name__)


@dataclass
class TextChunk:
    content: str
    chunk_index: int
    page_number: Optional[int]
    token_count: int


def estimate_tokens(text: str) -> int:
    # Rough estimate: 1 token ≈ 4 chars
    return len(text) // 4


def chunk_text(
    text: str,
    chunk_size: int = settings.CHUNK_SIZE,
    overlap: int = settings.CHUNK_OVERLAP,
    start_index: int = 0,
    page_number: Optional[int] = None,
) -> list[TextChunk]:
    """Split text into overlapping chunks by sentence boundary."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current_tokens = 0
    current_sentences: list[str] = []
    chunk_idx = start_index

    for sentence in sentences:
        s_tokens = estimate_tokens(sentence)
        if current_tokens + s_tokens > chunk_size and current_sentences:
            chunk_text_str = " ".join(current_sentences)
            chunks.append(TextChunk(
                content=chunk_text_str.strip(),
                chunk_index=chunk_idx,
                page_number=page_number,
                token_count=current_tokens,
            ))
            chunk_idx += 1
            # Overlap: keep last N tokens worth of sentences
            overlap_sentences = []
            overlap_tokens = 0
            for s in reversed(current_sentences):
                st = estimate_tokens(s)
                if overlap_tokens + st <= overlap:
                    overlap_sentences.insert(0, s)
                    overlap_tokens += st
                else:
                    break
            current_sentences = overlap_sentences + [sentence]
            current_tokens = overlap_tokens + s_tokens
        else:
            current_sentences.append(sentence)
            current_tokens += s_tokens

    if current_sentences:
        chunks.append(TextChunk(
            content=" ".join(current_sentences).strip(),
            chunk_index=chunk_idx,
            page_number=page_number,
            token_count=current_tokens,
        ))

    return chunks


def chunk_document(doc: "ParsedDocument") -> list[TextChunk]:
    """Chunk per-page to preserve page number metadata."""
    all_chunks: list[TextChunk] = []
    idx = 0

    for page in doc.pages:
        if len(page.text.strip()) < 50:
            continue
        page_chunks = chunk_text(
            page.text,
            start_index=idx,
            page_number=page.page_number,
        )
        all_chunks.extend(page_chunks)
        idx += len(page_chunks)

    logger.info("document_chunked", chunks=len(all_chunks))
    return all_chunks
