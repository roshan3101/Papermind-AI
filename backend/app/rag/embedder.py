import numpy as np
import google.generativeai as genai

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

GEMINI_EMBED_MODEL = "models/text-embedding-004"
EMBED_DIM = 768  # text-embedding-004 output dimension


def _init_genai():
    genai.configure(api_key=settings.GEMINI_API_KEY)


def get_embedding_model():
    """No-op — embeddings are API-based, nothing to load into memory."""
    _init_genai()
    logger.info("embedding_ready", model=GEMINI_EMBED_MODEL)
    return None


def embed_texts(texts: list[str], batch_size: int = 32) -> np.ndarray:
    _init_genai()
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        result = genai.embed_content(
            model=GEMINI_EMBED_MODEL,
            content=batch,
            task_type="retrieval_document",
        )
        all_embeddings.extend(result["embedding"])
    arr = np.array(all_embeddings, dtype=np.float32)
    # Normalize for cosine similarity
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, 1e-9)


def embed_query(query: str) -> np.ndarray:
    _init_genai()
    result = genai.embed_content(
        model=GEMINI_EMBED_MODEL,
        content=query,
        task_type="retrieval_query",
    )
    vec = np.array(result["embedding"], dtype=np.float32)
    vec = vec / max(np.linalg.norm(vec), 1e-9)
    return vec
