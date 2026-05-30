import numpy as np
from sentence_transformers import SentenceTransformer
from functools import lru_cache
from typing import Union

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    logger.info("loading_embedding_model", model=settings.EMBEDDING_MODEL)
    model = SentenceTransformer(settings.EMBEDDING_MODEL)
    logger.info("embedding_model_loaded")
    return model


def embed_texts(texts: list[str], batch_size: int = 32) -> np.ndarray:
    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,  # cosine similarity ready
    )
    return embeddings.astype(np.float32)


def embed_query(query: str) -> np.ndarray:
    # BGE models benefit from instruction prefix for retrieval
    prefixed = f"Represent this sentence for searching relevant passages: {query}"
    return embed_texts([prefixed])[0]
