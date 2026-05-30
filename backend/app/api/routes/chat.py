from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.core.security import get_current_user
from app.models.user import User
from app.schemas.paper import (
    ChatSessionOut, ChatSessionCreate, ChatAskRequest,
    ChatMessageOut, QAResponse,
)
from app.services.chat_service import ChatService

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/sessions", response_model=ChatSessionOut, status_code=201)
async def create_session(
    data: ChatSessionCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    session = await ChatService(db).create_session(current_user.id, data.paper_ids, data.title)
    return ChatSessionOut.model_validate({**session.__dict__, "messages": []})


@router.get("/sessions", response_model=list[ChatSessionOut])
async def list_sessions(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    sessions = await ChatService(db).list_sessions(current_user.id)
    return [ChatSessionOut.model_validate({**s.__dict__, "messages": []}) for s in sessions]


@router.get("/sessions/{session_id}", response_model=ChatSessionOut)
async def get_session(
    session_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    svc = ChatService(db)
    session = await svc.get_session(session_id, current_user.id)
    messages = await svc.get_messages(session_id, current_user.id)
    return ChatSessionOut.model_validate({**session.__dict__, "messages": messages})


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_session(
    session_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    await ChatService(db).delete_session(session_id, current_user.id)


@router.post("/ask", response_model=QAResponse)
async def chat_ask(
    req: ChatAskRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return await ChatService(db).ask(req.session_id, req.question, current_user.id, req.top_k)
