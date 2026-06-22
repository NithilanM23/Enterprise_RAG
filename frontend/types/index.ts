// types/index.ts — shared TypeScript types across the frontend

export interface Session {
  id: number;
  title: string;
  message_count: number;
  created_at: string;
  updated_at: string;
}

export interface Source {
  filename: string;
  chunk_number: number;
  chunk_text: string;
  reranker_score: number;
  similarity: number;
}

export interface Message {
  id: number;
  session_id: number;
  role: 'user' | 'assistant';
  content: string;
  sources?: Source[];
  is_pinned?: boolean;
  created_at: string;
}

export interface Document {
  id: number;
  filename: string;
  category: string;
  upload_time: string;
  chunk_count?: number;
  excel_row_count?: number;
  embedding_status?: string;
}

export interface DocumentStatus {
  doc_id: number;
  filename: string;
  stage: 'chunking' | 'embedding' | 'ready' | 'failed';
  chunks_total: number;
  chunks_embedded: number;
  excel_rows: number;
  percent_complete: number;
  error: string | null;
}

export interface Category {
  id: number;
  name: string;
  label: string;
  keywords: string[];
  weight: number;
  is_custom: boolean;
  document_count?: number;
}

export interface KnowledgeHealth {
  category: string;
  document_count: number;
  total_chunks: number;
  embedded: number;
  percent: number;
}

export interface PinnedAnswer {
  pin_id: number;
  message_id: number;
  note: string | null;
  pinned_at: string;
  content: string;
  sources: Source[];
  session_id: number;
  session_title: string;
}

export interface Settings {
  llm_model: string;
  embedding_model: string;
  reranker_model: string;
  chunk_size: number;
  chunk_overlap: number;
  top_k: number;
  semantic_k: number;
  bm25_k: number;
  mmr_pool: number;
  mmr_lambda: number;
  rrf_k: number;
  history_window: number;
  num_predict: number;
  temperature: number;
  routing_confidence_threshold: number;
}

export interface QueueStatus {
  active: number;
  queued: number;
  max_parallel: number;
  estimated_wait_seconds: number;
}

// SSE event types from /api/chat/ask
export type SSEEvent =
  | { type: 'routing'; category: string; confidence: number; scoped: boolean }
  | { type: 'sources'; chunks: Source[] }
  | { type: 'token'; text: string }
  | { type: 'done'; full_answer: string; session_id: number; message_id: number }
  | { type: 'followups'; suggestions: string[] }
  | { type: 'error'; message: string };

export interface SearchResult {
  session_id: number;
  title: string;
  snippet: string;
  matched_in: 'title' | 'message';
  updated_at: string;
}
