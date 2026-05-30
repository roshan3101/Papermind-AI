import re
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from typing import Optional

from app.core.logging import get_logger
from app.models.paper import Paper, PaperChunk
from app.models.query import QueryHistory
from app.rag.embedder import embed_query, embed_texts
from app.rag.vector_store import get_faiss_store
from app.rag.llm_client import (
    chat_completion,
    build_qa_messages,
    build_summary_messages,
    build_compare_messages,
)
from app.schemas.paper import (
    CitationSource, QAResponse, SummaryResponse,
    CompareResponse, SearchResult, SearchResponse,
    RecommendResponse,
)

logger = get_logger(__name__)


class RAGService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def _get_paper(self, paper_id: int, user_id: int) -> Paper:
        result = await self.db.execute(
            select(Paper).where(and_(Paper.id == paper_id, Paper.user_id == user_id))
        )
        paper = result.scalar_one_or_none()
        if not paper:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="Paper not found")
        return paper

    async def answer_question(
        self, question: str, user_id: int, paper_ids: Optional[list[int]] = None, top_k: int = 5
    ) -> QAResponse:
        store = get_faiss_store()
        q_vec = embed_query(question)
        hits = store.search(q_vec, top_k=top_k, paper_ids=paper_ids)

        if not hits:
            return QAResponse(answer="No relevant content found in the selected papers.", citations=[], tokens_used=0)

        faiss_ids = [h["faiss_id"] for h in hits]
        score_map = {h["faiss_id"]: h["score"] for h in hits}

        chunks_result = await self.db.execute(
            select(PaperChunk).where(PaperChunk.faiss_vector_id.in_(faiss_ids))
        )
        chunks = {c.faiss_vector_id: c for c in chunks_result.scalars().all()}

        paper_ids_needed = list({c.paper_id for c in chunks.values()})
        papers_result = await self.db.execute(
            select(Paper).where(and_(Paper.id.in_(paper_ids_needed), Paper.user_id == user_id))
        )
        papers = {p.id: p for p in papers_result.scalars().all()}

        context_chunks = []
        citations = []
        for fid in faiss_ids:
            chunk = chunks.get(fid)
            if not chunk:
                continue
            paper = papers.get(chunk.paper_id)
            if not paper:
                continue
            context_chunks.append({
                "paper_title": paper.title,
                "content": chunk.content,
                "page_number": chunk.page_number,
            })
            citations.append(CitationSource(
                paper_id=paper.id,
                paper_title=paper.title,
                chunk_index=chunk.chunk_index,
                page_number=chunk.page_number,
                content=chunk.content[:300],
                relevance_score=score_map.get(fid, 0.0),
            ))

        messages = build_qa_messages(question, context_chunks)
        answer, tokens = await chat_completion(messages)

        await self.db.execute(
            QueryHistory.__table__.insert().values(
                user_id=user_id,
                query=question,
                answer=answer,
                query_type="qa",
                paper_ids=paper_ids,
                citations=[c.model_dump() for c in citations],
                tokens_used=tokens,
            )
        )

        return QAResponse(answer=answer, citations=citations, tokens_used=tokens)

    async def summarize_paper(self, paper_id: int, user_id: int) -> SummaryResponse:
        paper = await self._get_paper(paper_id, user_id)

        if paper.summary:
            return self._parse_summary_response(paper, paper.summary)

        chunks_result = await self.db.execute(
            select(PaperChunk).where(PaperChunk.paper_id == paper_id).order_by(PaperChunk.chunk_index).limit(40)
        )
        chunks = chunks_result.scalars().all()
        full_text = "\n\n".join(c.content for c in chunks)

        messages = build_summary_messages(full_text, paper.title)
        summary_text, tokens = await chat_completion(messages, max_tokens=1500)

        paper.summary = summary_text
        await self.db.flush()

        await self.db.execute(
            QueryHistory.__table__.insert().values(
                user_id=user_id,
                query=f"Summarize: {paper.title}",
                answer=summary_text,
                query_type="summary",
                paper_ids=[paper_id],
                tokens_used=tokens,
            )
        )

        return self._parse_summary_response(paper, summary_text)

    def _parse_summary_response(self, paper: Paper, text: str) -> SummaryResponse:
        def extract_section(tag: str) -> str:
            pattern = rf"{tag}[:\s]*(.*?)(?=\n\d+\.|$)"
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            return match.group(1).strip() if match else ""

        summary = extract_section("SUMMARY") or text[:500]
        methodology = extract_section("METHODOLOGY")
        results = extract_section("RESULTS")
        limitations = extract_section("LIMITATIONS")

        contributions_block = extract_section("KEY_CONTRIBUTIONS")
        contributions = [
            l.strip("•- ").strip()
            for l in contributions_block.split("\n")
            if l.strip("•- ").strip()
        ] or [contributions_block]

        return SummaryResponse(
            paper_id=paper.id,
            title=paper.title,
            summary=summary,
            key_contributions=contributions,
            methodology=methodology,
            results=results,
            limitations=limitations,
        )

    async def compare_papers(self, paper_id_1: int, paper_id_2: int, user_id: int) -> CompareResponse:
        paper1 = await self._get_paper(paper_id_1, user_id)
        paper2 = await self._get_paper(paper_id_2, user_id)

        async def get_text(pid: int) -> str:
            r = await self.db.execute(
                select(PaperChunk).where(PaperChunk.paper_id == pid).order_by(PaperChunk.chunk_index).limit(30)
            )
            return "\n\n".join(c.content for c in r.scalars().all())

        text1 = await get_text(paper_id_1)
        text2 = await get_text(paper_id_2)

        messages = build_compare_messages(text1, paper1.title, text2, paper2.title)
        comparison_text, tokens = await chat_completion(messages, max_tokens=2000)

        def extract(tag: str) -> str:
            pattern = rf"{tag}[:\s]*(.*?)(?=\n\d+\.|$)"
            m = re.search(pattern, comparison_text, re.IGNORECASE | re.DOTALL)
            return m.group(1).strip() if m else ""

        await self.db.execute(
            QueryHistory.__table__.insert().values(
                user_id=user_id,
                query=f"Compare: {paper1.title} vs {paper2.title}",
                answer=comparison_text,
                query_type="compare",
                paper_ids=[paper_id_1, paper_id_2],
                tokens_used=tokens,
            )
        )

        return CompareResponse(
            paper_1_title=paper1.title,
            paper_2_title=paper2.title,
            methodology=extract("METHODOLOGY"),
            datasets=extract("DATASETS"),
            performance=extract("PERFORMANCE_METRICS"),
            conclusions=extract("CONCLUSIONS"),
            overall_comparison=extract("OVERALL_COMPARISON") or comparison_text[:1000],
        )

    async def semantic_search(
        self,
        query: str,
        user_id: int,
        author: Optional[str] = None,
        year: Optional[int] = None,
        top_k: int = 10,
    ) -> SearchResponse:
        store = get_faiss_store()
        q_vec = embed_query(query)
        hits = store.search(q_vec, top_k=top_k * 3)

        if not hits:
            return SearchResponse(results=[], total=0)

        faiss_ids = [h["faiss_id"] for h in hits]
        score_map = {h["faiss_id"]: h["score"] for h in hits}

        chunks_result = await self.db.execute(
            select(PaperChunk).where(PaperChunk.faiss_vector_id.in_(faiss_ids))
        )
        chunks = {c.faiss_vector_id: c for c in chunks_result.scalars().all()}

        paper_q = select(Paper).where(and_(Paper.user_id == user_id))
        if author:
            paper_q = paper_q.where(Paper.authors.ilike(f"%{author}%"))
        if year:
            paper_q = paper_q.where(Paper.year == year)
        papers_result = await self.db.execute(paper_q)
        papers = {p.id: p for p in papers_result.scalars().all()}

        results = []
        for fid in faiss_ids:
            chunk = chunks.get(fid)
            if not chunk or chunk.paper_id not in papers:
                continue
            paper = papers[chunk.paper_id]
            results.append(SearchResult(
                paper_id=paper.id,
                paper_title=paper.title,
                authors=paper.authors,
                year=paper.year,
                chunk_content=chunk.content[:500],
                page_number=chunk.page_number,
                relevance_score=score_map.get(fid, 0.0),
            ))
            if len(results) >= top_k:
                break

        await self.db.execute(
            QueryHistory.__table__.insert().values(
                user_id=user_id,
                query=query,
                query_type="search",
                tokens_used=0,
            )
        )

        return SearchResponse(results=results, total=len(results))

    async def recommend_related(self, paper_id: int, user_id: int, top_k: int = 5) -> RecommendResponse:
        paper = await self._get_paper(paper_id, user_id)

        # Get representative chunks from source paper
        chunks_result = await self.db.execute(
            select(PaperChunk).where(PaperChunk.paper_id == paper_id).limit(5)
        )
        source_chunks = chunks_result.scalars().all()
        if not source_chunks:
            return RecommendResponse(recommendations=[])

        texts = [c.content for c in source_chunks]
        embeddings = embed_texts(texts)
        avg_embedding = embeddings.mean(axis=0)

        store = get_faiss_store()
        hits = store.search(avg_embedding, top_k=top_k * 5)

        seen_paper_ids = {paper_id}
        recommendations = []

        for hit in hits:
            pid = hit.get("paper_id")
            if pid in seen_paper_ids:
                continue
            seen_paper_ids.add(pid)

            p_result = await self.db.execute(
                select(Paper).where(and_(Paper.id == pid, Paper.user_id == user_id))
            )
            p = p_result.scalar_one_or_none()
            if not p:
                continue
            recommendations.append({
                "paper_id": p.id,
                "title": p.title,
                "authors": p.authors,
                "year": p.year,
                "similarity_score": round(hit["score"], 4),
            })
            if len(recommendations) >= top_k:
                break

        return RecommendResponse(recommendations=recommendations)
