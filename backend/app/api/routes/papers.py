from fastapi import APIRouter, Depends, UploadFile, File, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

from app.db.session import get_db
from app.core.security import get_current_user
from app.models.user import User
from app.schemas.paper import PaperOut, PaperListOut, DashboardStats
from app.services.paper_service import PaperService

router = APIRouter(prefix="/papers", tags=["papers"])


@router.post("/upload", response_model=PaperOut, status_code=201)
async def upload_paper(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = PaperService(db)
    return await service.upload_and_process(file, current_user.id)


@router.get("", response_model=PaperListOut)
async def list_papers(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    author: Optional[str] = Query(None),
    year: Optional[int] = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = PaperService(db)
    papers, total = await service.get_user_papers(current_user.id, skip, limit, author, year)
    return PaperListOut(papers=[PaperOut.model_validate(p) for p in papers], total=total)


@router.get("/dashboard", response_model=DashboardStats)
async def get_dashboard(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return await PaperService(db).get_dashboard_stats(current_user.id)


@router.get("/{paper_id}", response_model=PaperOut)
async def get_paper(
    paper_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return await PaperService(db).get_paper(paper_id, current_user.id)


@router.delete("/{paper_id}", status_code=204)
async def delete_paper(
    paper_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    await PaperService(db).delete_paper(paper_id, current_user.id)
