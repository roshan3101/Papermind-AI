from sqlalchemy import String, Text, Integer, ForeignKey, DateTime, func, JSON, Float
from sqlalchemy.orm import Mapped, mapped_column, relationship
from datetime import datetime
from typing import Optional

from app.db.session import Base


class Paper(Base):
    __tablename__ = "papers"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    authors: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    abstract: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    year: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(String(1000), nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    page_count: Mapped[int] = mapped_column(Integer, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    faiss_index_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    status: Mapped[str] = mapped_column(String(50), default="processing")  # processing, ready, error
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    paper_metadata: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    owner: Mapped["User"] = relationship("User", back_populates="papers")
    chunks: Mapped[list["PaperChunk"]] = relationship("PaperChunk", back_populates="paper", cascade="all, delete-orphan")


class PaperChunk(Base):
    __tablename__ = "paper_chunks"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    paper_id: Mapped[int] = mapped_column(ForeignKey("papers.id", ondelete="CASCADE"), nullable=False, index=True)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    page_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    faiss_vector_id: Mapped[int] = mapped_column(Integer, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    paper: Mapped["Paper"] = relationship("Paper", back_populates="chunks")
