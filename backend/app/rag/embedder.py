import numpy as np
from google import genai
from google.genai import types
from functools import lru_cache

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 3072


@lru_cache(maxsize=1)
def _client() -> genai.Client:
    # text-embedding-004 is only available on v1, not v1beta
    return genai.Client(api_key=settings.GEMINI_API_KEY, http_options={"api_version": "v1"})


def get_embedding_model():
    """No-op — embeddings are API-based."""
    _client()
    logger.info("embedding_ready", model=EMBED_MODEL)
    return None


def embed_texts(texts: list[str], batch_size: int = 32) -> np.ndarray:
    client = _client()
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.models.embed_content(
            model=EMBED_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )
        for emb in response.embeddings:
            all_embeddings.append(emb.values)
    arr = np.array(all_embeddings, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, 1e-9)


def embed_query(query: str) -> np.ndarray:
    client = _client()
    response = client.models.embed_content(
        model=EMBED_MODEL,
        contents=query,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    vec = np.array(response.embeddings[0].values, dtype=np.float32)
    return vec / max(np.linalg.norm(vec), 1e-9)
