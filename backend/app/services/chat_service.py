import uuid
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from typing import Optional
from fastapi import HTTPException

from app.core.logging import get_logger
from app.models.query import ChatSession, ChatMessage
from app.models.paper import Paper, PaperChunk
from app.rag.embedder import embed_query
from app.rag.vector_store import get_faiss_store
from app.rag.llm_client import chat_completion, build_qa_messages
from app.schemas.paper import CitationSource, QAResponse

logger = get_logger(__name__)

# How many past exchanges to include as history
HISTORY_WINDOW = 6


class ChatService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_session(self, user_id: int, paper_ids: Optional[list[int]], title: str) -> ChatSession:
        session = ChatSession(
            id=str(uuid.uuid4()),
            user_id=user_id,
            title=title,
            paper_ids=paper_ids,
        )
        self.db.add(session)
        await self.db.flush()
        await self.db.refresh(session)
        return session

    async def get_session(self, session_id: str, user_id: int) -> ChatSession:
        result = await self.db.execute(
            select(ChatSession).where(and_(ChatSession.id == session_id, ChatSession.user_id == user_id))
        )
        session = result.scalar_one_or_none()
        if not session:
            raise HTTPException(status_code=404, detail="Chat session not found")
        return session

    async def list_sessions(self, user_id: int) -> list[ChatSession]:
        result = await self.db.execute(
            select(ChatSession)
            .where(ChatSession.user_id == user_id)
            .order_by(ChatSession.updated_at.desc())
        )
        return list(result.scalars().all())

    async def get_messages(self, session_id: str, user_id: int) -> list[ChatMessage]:
        await self.get_session(session_id, user_id)  # auth check
        result = await self.db.execute(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at)
        )
        return list(result.scalars().all())

    async def delete_session(self, session_id: str, user_id: int):
        session = await self.get_session(session_id, user_id)
        await self.db.delete(session)
        await self.db.flush()

    async def ask(self, session_id: str, question: str, user_id: int, top_k: int = 5) -> QAResponse:
        session = await self.get_session(session_id, user_id)

        # ── RAG retrieval ──────────────────────────────────────────────────
        store = get_faiss_store()
        q_vec = embed_query(question)
        paper_ids = session.paper_ids or None
        hits = store.search(q_vec, top_k=top_k, paper_ids=paper_ids)

        context_chunks: list[dict] = []
        citations: list[CitationSource] = []

        if hits:
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

        # ── Build messages with conversation history ───────────────────────
        history_result = await self.db.execute(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.desc())
            .limit(HISTORY_WINDOW)
        )
        past_messages = list(reversed(history_result.scalars().all()))

        messages = _build_chat_messages(question, context_chunks, past_messages)
        answer, tokens = await chat_completion(messages)

        # ── Persist user message + assistant reply ─────────────────────────
        user_msg = ChatMessage(session_id=session_id, role="user", content=question)
        assistant_msg = ChatMessage(
            session_id=session_id,
            role="assistant",
            content=answer,
            citations=[c.model_dump() for c in citations],
            tokens_used=tokens,
        )
        self.db.add_all([user_msg, assistant_msg])

        # Update session title from first question
        if not past_messages:
            session.title = question[:120]

        await self.db.flush()

        return QAResponse(answer=answer, citations=citations, tokens_used=tokens)


def _build_chat_messages(
    question: str,
    context_chunks: list[dict],
    history: list[ChatMessage],
) -> list[dict]:
    system = (
        "You are PaperMind AI, an expert research assistant. "
        "Answer questions based ONLY on the provided research paper excerpts. "
        "Do NOT include inline citations like [Paper: ...] or [Source ...] in your response — citations are shown separately. "
        "Write clean, well-structured answers using markdown (headings, bullet points, bold) where appropriate. "
        "You have access to the conversation history — use it to give coherent, context-aware answers. "
        "If the context doesn't have enough information, say so clearly."
    )

    context_block = "\n\n".join(
        f"[Source {i+1}: {c['paper_title']}, Page {c.get('page_number', 'N/A')}]\n{c['content']}"
        for i, c in enumerate(context_chunks)
    )

    messages: list[dict] = [{"role": "system", "content": system}]

    # Inject past conversation turns
    for msg in history:
        messages.append({"role": msg.role, "content": msg.content})

    # Current question with fresh context
    user_content = f"Context from papers:\n{context_block}\n\nQuestion: {question}" if context_block else question
    messages.append({"role": "user", "content": user_content})

    return messages
