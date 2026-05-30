export interface User {
  id: number
  email: string
  username: string
  is_active: boolean
  created_at: string
}

export interface AuthResponse {
  access_token: string
  token_type: string
  user: User
}

export interface Paper {
  id: number
  title: string
  authors: string | null
  abstract: string | null
  year: number | null
  filename: string
  file_size: number
  page_count: number
  chunk_count: number
  status: 'processing' | 'ready' | 'error'
  summary: string | null
  created_at: string
}

export interface CitationSource {
  paper_id: number
  paper_title: string
  chunk_index: number
  page_number: number | null
  content: string
  relevance_score: number
}

export interface QAResponse {
  answer: string
  citations: CitationSource[]
  tokens_used: number
}

export interface SummaryResponse {
  paper_id: number
  title: string
  summary: string
  key_contributions: string[]
  methodology: string
  results: string
  limitations: string
}

export interface CompareResponse {
  paper_1_title: string
  paper_2_title: string
  methodology: string
  datasets: string
  performance: string
  conclusions: string
  overall_comparison: string
}

export interface SearchResult {
  paper_id: number
  paper_title: string
  authors: string | null
  year: number | null
  chunk_content: string
  page_number: number | null
  relevance_score: number
}

export interface SearchResponse {
  results: SearchResult[]
  total: number
}

export interface Recommendation {
  paper_id: number
  title: string
  authors: string | null
  year: number | null
  similarity_score: number
}

export interface DashboardStats {
  total_papers: number
  total_chunks: number
  total_queries: number
  recent_queries: Array<{ id: number; query: string; type: string; created_at: string }>
  papers_by_status: Record<string, number>
}
