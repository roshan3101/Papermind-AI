from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.core.security import get_current_user
from app.models.user import User
from app.schemas.paper import (
    QARequest, QAResponse,
    SummaryRequest, SummaryResponse,
    CompareRequest, CompareResponse,
    SearchRequest, SearchResponse,
    RecommendRequest, RecommendResponse,
)
from app.services.rag_service import RAGService

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/ask", response_model=QAResponse)
async def ask_question(
    req: QARequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return await RAGService(db).answer_question(
        req.question, current_user.id, req.paper_ids, req.top_k
    )


@router.post("/summarize", response_model=SummaryResponse)
async def summarize(
    req: SummaryRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return await RAGService(db).summarize_paper(req.paper_id, current_user.id)


@router.post("/compare", response_model=CompareResponse)
async def compare(
    req: CompareRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return await RAGService(db).compare_papers(req.paper_id_1, req.paper_id_2, current_user.id)


@router.post("/search", response_model=SearchResponse)
async def search(
    req: SearchRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return await RAGService(db).semantic_search(
        req.query, current_user.id, req.author, req.year, req.top_k
    )


@router.post("/recommend", response_model=RecommendResponse)
async def recommend(
    req: RecommendRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return await RAGService(db).recommend_related(req.paper_id, current_user.id, req.top_k)
