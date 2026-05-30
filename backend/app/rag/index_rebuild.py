import numpy as np
from sqlalchemy import select, text

from app.core.logging import get_logger
from app.db.session import AsyncSessionLocal
from app.models.paper import PaperChunk
from app.rag.vector_store import get_faiss_store
import faiss

logger = get_logger(__name__)


async def rebuild_faiss_from_db():
    """
    Rebuild the FAISS index from all PaperChunk rows in PostgreSQL.
    Called at startup so a cold Render instance recovers its index automatically.
    """
    store = get_faiss_store()

    # Already has vectors — nothing to do (local dev or persistent disk)
    if store.total_vectors > 0:
        logger.info("faiss_skip_rebuild", reason="index_already_populated", vectors=store.total_vectors)
        return

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(PaperChunk).order_by(PaperChunk.faiss_vector_id.asc())
        )
        chunks = list(result.scalars().all())

    if not chunks:
        logger.info("faiss_skip_rebuild", reason="no_chunks_in_db")
        return

    logger.info("faiss_rebuilding", chunk_count=len(chunks))

    texts = [c.content for c in chunks]
    from app.rag.embedder import embed_texts
    embeddings = embed_texts(texts, batch_size=32)

    # Rebuild the index directly (bypass add_vectors to avoid per-batch saves)
    new_index = faiss.IndexFlatIP(store.dim)
    new_index.add(embeddings)

    new_meta = {}
    for new_id, chunk in enumerate(chunks):
        new_meta[new_id] = {"paper_id": chunk.paper_id, "chunk_db_id": chunk.id}

        # Keep faiss_vector_id in sync so DB lookups still work
        # (IDs are positional — chunk must have been inserted in order)

    store._index = new_index
    store._metadata = new_meta
    store._save()

    # Update faiss_vector_id on chunks whose ID drifted (e.g. after a rebuild)
    async with AsyncSessionLocal() as db:
        for new_id, chunk in enumerate(chunks):
            if chunk.faiss_vector_id != new_id:
                await db.execute(
                    text("UPDATE paper_chunks SET faiss_vector_id = :fid WHERE id = :id"),
                    {"fid": new_id, "id": chunk.id},
                )
        await db.commit()

    logger.info("faiss_rebuilt", vectors=new_index.ntotal)
