import os
import re
from pathlib import Path
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_
from fastapi import HTTPException, UploadFile, status

from app.core.config import settings
from app.core.logging import get_logger
from app.models.paper import Paper, PaperChunk
from app.models.query import QueryHistory
from app.rag.pdf_parser import parse_pdf
from app.rag.chunker import chunk_document
from app.rag.embedder import embed_texts
from app.rag.vector_store import get_faiss_store
from app.utils.cloudinary_client import upload_pdf, download_pdf_to_temp, delete_pdf

logger = get_logger(__name__)


class PaperService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def upload_and_process(self, file: UploadFile, user_id: int) -> Paper:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are accepted")

        contents = await file.read()
        if len(contents) > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(status_code=413, detail=f"File exceeds {settings.MAX_FILE_SIZE_MB}MB limit")

        safe_name = re.sub(r"[^\w\-_. ]", "_", file.filename)

        # Upload to Cloudinary
        cloud = upload_pdf(contents, safe_name, user_id)

        paper = Paper(
            user_id=user_id,
            title=safe_name,
            filename=safe_name,
            file_path=cloud["url"],           # Cloudinary secure URL
            file_size=len(contents),
            status="processing",
            paper_metadata={"cloudinary_public_id": cloud["public_id"]},
        )
        self.db.add(paper)
        await self.db.flush()
        await self.db.refresh(paper)

        try:
            await self._process_paper(paper, contents)
        except Exception as e:
            logger.error("paper_processing_failed", paper_id=paper.id, error=str(e))
            paper.status = "error"
            await self.db.flush()

        return paper

    async def _process_paper(self, paper: Paper, pdf_bytes: bytes):
        """Parse directly from bytes (no re-download needed since we just uploaded)."""
        import tempfile

        # Write bytes to temp file for PyMuPDF (it needs a file path)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp.write(pdf_bytes)
        tmp.close()

        try:
            parsed = parse_pdf(tmp.name)
        finally:
            os.unlink(tmp.name)

        paper.title = parsed.title
        paper.authors = parsed.authors
        paper.abstract = parsed.abstract
        paper.year = parsed.year
        paper.page_count = parsed.page_count

        # Preserve cloudinary metadata
        existing_meta = paper.paper_metadata or {}
        paper.paper_metadata = {**existing_meta, **parsed.metadata}

        chunks = chunk_document(parsed)
        texts = [c.content for c in chunks]
        embeddings = embed_texts(texts)

        store = get_faiss_store()
        meta_list = [{"paper_id": paper.id, "chunk_index": c.chunk_index} for c in chunks]
        faiss_ids = store.add_vectors(embeddings, meta_list)

        db_chunks = [
            PaperChunk(
                paper_id=paper.id,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                page_number=chunk.page_number,
                faiss_vector_id=fid,
                token_count=chunk.token_count,
            )
            for chunk, fid in zip(chunks, faiss_ids)
        ]
        self.db.add_all(db_chunks)
        paper.chunk_count = len(chunks)
        paper.status = "ready"
        await self.db.flush()
        logger.info("paper_processed", paper_id=paper.id, chunks=len(chunks))

    async def get_user_papers(
        self,
        user_id: int,
        skip: int = 0,
        limit: int = 50,
        author: Optional[str] = None,
        year: Optional[int] = None,
    ) -> tuple[list[Paper], int]:
        q = select(Paper).where(Paper.user_id == user_id)
        if author:
            q = q.where(Paper.authors.ilike(f"%{author}%"))
        if year:
            q = q.where(Paper.year == year)

        count_q = select(func.count()).select_from(q.subquery())
        total = (await self.db.execute(count_q)).scalar_one()
        q = q.order_by(Paper.created_at.desc()).offset(skip).limit(limit)
        papers = (await self.db.execute(q)).scalars().all()
        return list(papers), total

    async def get_paper(self, paper_id: int, user_id: int) -> Paper:
        result = await self.db.execute(
            select(Paper).where(and_(Paper.id == paper_id, Paper.user_id == user_id))
        )
        paper = result.scalar_one_or_none()
        if not paper:
            raise HTTPException(status_code=404, detail="Paper not found")
        return paper

    async def delete_paper(self, paper_id: int, user_id: int):
        paper = await self.get_paper(paper_id, user_id)

        # Delete from Cloudinary
        meta = paper.paper_metadata or {}
        public_id = meta.get("cloudinary_public_id")
        if public_id:
            delete_pdf(public_id)

        # Delete from FAISS
        store = get_faiss_store()
        store.delete_by_paper(paper_id)

        await self.db.delete(paper)
        await self.db.flush()

    async def get_dashboard_stats(self, user_id: int) -> dict:
        total_papers = (await self.db.execute(
            select(func.count()).select_from(Paper).where(Paper.user_id == user_id)
        )).scalar_one()

        total_chunks = (await self.db.execute(
            select(func.sum(Paper.chunk_count)).where(Paper.user_id == user_id)
        )).scalar_one() or 0

        total_queries = (await self.db.execute(
            select(func.count()).select_from(QueryHistory).where(QueryHistory.user_id == user_id)
        )).scalar_one()

        recent_q = await self.db.execute(
            select(QueryHistory)
            .where(QueryHistory.user_id == user_id)
            .order_by(QueryHistory.created_at.desc())
            .limit(10)
        )
        recent_queries = [
            {"id": q.id, "query": q.query, "type": q.query_type, "created_at": str(q.created_at)}
            for q in recent_q.scalars().all()
        ]

        status_q = await self.db.execute(
            select(Paper.status, func.count()).where(Paper.user_id == user_id).group_by(Paper.status)
        )
        papers_by_status = {row[0]: row[1] for row in status_q.all()}

        return {
            "total_papers": total_papers,
            "total_chunks": int(total_chunks),
            "total_queries": total_queries,
            "recent_queries": recent_queries,
            "papers_by_status": papers_by_status,
        }
