from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logging import setup_logging, get_logger
from app.db.session import engine
from app.api.routes import auth, papers, rag, chat

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging(settings.DEBUG)
    logger.info("papermind_starting", version=settings.VERSION, provider=settings.LLM_PROVIDER)

    # Init FAISS store, rebuild from DB if empty (handles Render cold starts)
    from app.rag.vector_store import get_faiss_store
    from app.rag.index_rebuild import rebuild_faiss_from_db
    get_faiss_store()
    await rebuild_faiss_from_db()
    store = get_faiss_store()
    logger.info("faiss_ready", vectors=store.total_vectors)

    yield

    logger.info("papermind_shutting_down")
    await engine.dispose()


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="Citation-aware AI Research Assistant",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/api/v1")
app.include_router(papers.router, prefix="/api/v1")
app.include_router(rag.router, prefix="/api/v1")
app.include_router(chat.router, prefix="/api/v1")


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": settings.VERSION, "app": settings.APP_NAME}
