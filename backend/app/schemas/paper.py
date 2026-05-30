from pydantic import BaseModel
from datetime import datetime
from typing import Optional, Any


class PaperOut(BaseModel):
    id: int
    title: str
    authors: Optional[str]
    abstract: Optional[str]
    year: Optional[int]
    filename: str
    file_size: int
    page_count: int
    chunk_count: int
    status: str
    summary: Optional[str]
    created_at: datetime

    model_config = {"from_attributes": True}


class PaperListOut(BaseModel):
    papers: list[PaperOut]
    total: int


class CitationSource(BaseModel):
    paper_id: int
    paper_title: str
    chunk_index: int
    page_number: Optional[int]
    content: str
    relevance_score: float


class QARequest(BaseModel):
    question: str
    paper_ids: Optional[list[int]] = None  # None = search all user's papers
    top_k: int = 5


class QAResponse(BaseModel):
    answer: str
    citations: list[CitationSource]
    tokens_used: int


class SummaryRequest(BaseModel):
    paper_id: int


class SummaryResponse(BaseModel):
    paper_id: int
    title: str
    summary: str
    key_contributions: list[str]
    methodology: str
    results: str
    limitations: str


class CompareRequest(BaseModel):
    paper_id_1: int
    paper_id_2: int


class CompareResponse(BaseModel):
    paper_1_title: str
    paper_2_title: str
    methodology: str
    datasets: str
    performance: str
    conclusions: str
    overall_comparison: str


class SearchRequest(BaseModel):
    query: str
    author: Optional[str] = None
    year: Optional[int] = None
    top_k: int = 10


class SearchResult(BaseModel):
    paper_id: int
    paper_title: str
    authors: Optional[str]
    year: Optional[int]
    chunk_content: str
    page_number: Optional[int]
    relevance_score: float


class SearchResponse(BaseModel):
    results: list[SearchResult]
    total: int


class RecommendRequest(BaseModel):
    paper_id: int
    top_k: int = 5


class RecommendResponse(BaseModel):
    recommendations: list[dict]


class DashboardStats(BaseModel):
    total_papers: int
    total_chunks: int
    total_queries: int
    recent_queries: list[dict]
    papers_by_status: dict


class ChatMessageOut(BaseModel):
    id: int
    role: str
    content: str
    citations: Optional[list] = None
    tokens_used: int
    created_at: datetime

    model_config = {"from_attributes": True}


class ChatSessionOut(BaseModel):
    id: str
    title: str
    paper_ids: Optional[list] = None
    created_at: datetime
    updated_at: datetime
    messages: list[ChatMessageOut] = []

    model_config = {"from_attributes": True}


class ChatSessionCreate(BaseModel):
    paper_ids: Optional[list[int]] = None
    title: str = "New Chat"


class ChatAskRequest(BaseModel):
    session_id: str
    question: str
    top_k: int = 5
