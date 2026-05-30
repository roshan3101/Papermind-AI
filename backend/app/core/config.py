from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional, Literal


class Settings(BaseSettings):
    # App
    APP_NAME: str = "PaperMind AI"
    DEBUG: bool = False
    VERSION: str = "1.0.0"

    # Security
    SECRET_KEY: str = "change-me-in-production-use-secrets-module"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24

    # Database (Render PostgreSQL or local)
    DATABASE_URL: str = "postgresql+asyncpg://papermind:papermind@localhost:5432/papermind"
    DATABASE_URL_SYNC: str = "postgresql+psycopg2://papermind:papermind@localhost:5432/papermind"

    # ── LLM Provider ────────────────────────────────────────────────────────
    # Set LLM_PROVIDER to one of: openai | openrouter | gemini | anthropic
    LLM_PROVIDER: Literal["openai", "openrouter", "gemini", "anthropic"] = "openai"

    # OpenAI
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o-mini"

    # OpenRouter  (uses OpenAI-compatible base URL)
    OPENROUTER_API_KEY: Optional[str] = None
    OPENROUTER_MODEL: str = "meta-llama/llama-3.1-8b-instruct"

    # Google Gemini
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-1.5-flash"

    # Anthropic Claude
    ANTHROPIC_API_KEY: Optional[str] = None
    ANTHROPIC_MODEL: str = "claude-3-5-haiku-20241022"

    # Embeddings (Google text-embedding-004 via API — no local model needed)
    EMBEDDING_MODEL: str = "gemini-embedding-001"
    EMBEDDING_DIM: int = 3072

    # RAG
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64
    TOP_K_RETRIEVAL: int = 5
    FAISS_INDEX_PATH: str = "./faiss_index"

    # Upload
    MAX_FILE_SIZE_MB: int = 50

    # Cloudinary
    CLOUDINARY_CLOUD_NAME: str = ""
    CLOUDINARY_API_KEY: str = ""
    CLOUDINARY_API_SECRET: str = ""

    # CORS — set as comma-separated string on Render, e.g.: https://app.vercel.app,http://localhost:3000
    ALLOWED_ORIGINS: str = "http://localhost:3000,http://localhost:3001"

    @property
    def allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",") if o.strip()]

    class Config:
        env_file = ".env"
        extra = "allow"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
