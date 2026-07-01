import { getSession } from 'next-auth/react';

const API = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const session = await getSession();
  const token = (session as any)?.accessToken;
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  if (token) headers['Authorization'] = `Bearer ${token}`;

  const res = await fetch(`${API}${path}`, {
    headers: { ...headers, ...options?.headers },
    ...options,
  });
  if (!res.ok) {
    if (res.status === 401 && typeof window !== 'undefined') {
      window.location.href = '/api/auth/signin';
    }
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

// ── Auth ──────────────────────────────────────────────────────────────────

export const auth = {
  login: (data: FormData) => fetch(`${API}/api/auth/login`, { method: 'POST', body: data }).then(r => {
    if (!r.ok) throw new Error('Login failed');
    return r.json();
  }),
  register: (data: any) => request<any>('/api/auth/register', { method: 'POST', body: JSON.stringify(data) })
};

// ── Health ────────────────────────────────────────────────────────────────

export const health = {
  get:              () => request<any>('/api/health'),
  embeddings:       () => request<any[]>('/api/health/embeddings'),
  knowledgeHealth:  () => request<any[]>('/api/status/knowledge-health'),
};

// ── Documents ─────────────────────────────────────────────────────────────

export const documents = {
  list: () => request<any[]>('/api/documents'),

  upload: async (file: File, category: string, newCategoryLabel?: string) => {
    const fd = new FormData();
    fd.append('file', file);
    fd.append('category', category);
    if (newCategoryLabel) fd.append('new_category_label', newCategoryLabel);
    const session = await getSession();
    const token = (session as any)?.accessToken;
    const headers: Record<string, string> = {};
    if (token) headers['Authorization'] = `Bearer ${token}`;
    
    return fetch(`${API}/api/documents/upload`, { method: 'POST', headers, body: fd })
      .then(r => r.json());
  },

  status:   (id: number) => request<any>(`/api/documents/${id}/status`),
  get:      (id: number) => request<any>(`/api/documents/${id}`),
  metadata: (id: number) => request<any>(`/api/documents/${id}/metadata`),

  delete: (id: number) =>
    request<any>(`/api/documents/${id}`, { method: 'DELETE' }),

  setCategory: (id: number, category: string) =>
    request<any>(`/api/documents/${id}/category`, {
      method: 'PATCH',
      body: JSON.stringify({ category }),
    }),

  embed: () => request<any>('/api/documents/embed', { method: 'POST' }),
};

// ── Categories ────────────────────────────────────────────────────────────

export const categories = {
  list: () => request<any[]>('/api/categories'),

  create: (label: string, keywords?: string[], weight?: number) =>
    request<any>('/api/categories', {
      method: 'POST',
      body: JSON.stringify({ label, keywords, weight }),
    }),

  updateKeywords: (name: string, keywords: string[], weight?: number) =>
    request<any>(`/api/categories/${name}/keywords`, {
      method: 'PATCH',
      body: JSON.stringify({ keywords, weight }),
    }),

  delete: (name: string) =>
    request<any>(`/api/categories/${name}`, { method: 'DELETE' }),
};

// ── Sessions ──────────────────────────────────────────────────────────────

export const sessions = {
  list:    () => request<any[]>('/api/sessions'),
  create:  (title?: string, app_mode: string = 'rag') => request<any>('/api/sessions', { method: 'POST', body: JSON.stringify({ title: title || 'New Chat', app_mode }) }),
  messages: (id: number)   => request<any[]>(`/api/sessions/${id}/messages`),
  rename:  (id: number, title: string) => request<any>(`/api/sessions/${id}`, { method: 'PATCH', body: JSON.stringify({ title }) }),
  delete:  (id: number)    => request<any>(`/api/sessions/${id}`, { method: 'DELETE' }),
  search:  (q: string)     => request<any[]>(`/api/sessions/search?q=${encodeURIComponent(q)}`),
};

// ── Chat / SSE streaming ──────────────────────────────────────────────────

export function streamAsk(
  sessionId: number,
  question: string,
  documentIds: number[] | null,
  use_rag: boolean,
  onEvent: (event: any) => void,
  onDone: () => void,
  onError: (msg: string) => void
): AbortController {
  const controller = new AbortController();

  getSession().then((session) => {
    const token = (session as any)?.accessToken;
    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    if (token) headers['Authorization'] = `Bearer ${token}`;

    fetch(`${API}/api/chat/ask`, {
      method: 'POST',
      headers,
      body: JSON.stringify({ session_id: sessionId, question, document_ids: documentIds, use_rag }),
      signal: controller.signal,
    }).then(async (res) => {
      if (!res.ok) {
        onError(`Server error ${res.status}`);
        return;
      }
      const reader = res.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed || !trimmed.startsWith('data: ')) continue;
          try {
            const evt = JSON.parse(trimmed.slice(6));
            onEvent(evt);
            if (evt.type === 'done') onDone();
          } catch {}
        }
      }
    }).catch((err) => {
      if (err.name !== 'AbortError') onError(err.message);
    });
  });

  return controller;
}

// ── Saved / Pinned ────────────────────────────────────────────────────────

export const saved = {
  list:   () => request<any[]>('/api/saved'),
  pin:    (messageId: number, note?: string) =>
    request<any>(`/api/messages/${messageId}/pin`, { method: 'POST', body: JSON.stringify({ note }) }),
  unpin:  (messageId: number) =>
    request<any>(`/api/messages/${messageId}/pin`, { method: 'DELETE' }),
};

// ── Data Explorer ─────────────────────────────────────────────────────────

export const explorer = {
  upload: async (file: File) => {
    const fd = new FormData();
    fd.append('file', file);
    const session = await getSession();
    const token = (session as any)?.accessToken;
    const headers: Record<string, string> = {};
    if (token) headers['Authorization'] = `Bearer ${token}`;
    return fetch(`${API}/api/explorer/upload`, { method: 'POST', headers, body: fd }).then(r => r.json());
  },
  query: (token: string, query: string, filename: string) =>
    request<any>('/api/explorer/query', {
      method: 'POST',
      body: JSON.stringify({ token, query, filename }),
    }),
  schema: (token: string) => request<any>(`/api/explorer/schema/${token}`),
};

// ── Admin ────────────────────────────────────────────────────────────────

export const admin = {
  getSettings:       () => request<any>('/api/admin/settings'),
  updateSettings:    (body: Partial<any>) => request<any>('/api/admin/settings', { method: 'PATCH', body: JSON.stringify(body) }),
  resetSetting:      (key: string) => request<any>(`/api/admin/settings/reset/${key}`, { method: 'POST' }),
  listModels:        () => request<any>('/api/admin/models'),
  swapLLM:           (model: string) => request<any>('/api/admin/settings/llm-model', { method: 'POST', body: JSON.stringify({ model }) }),
  swapReranker:      (model: string) => request<any>('/api/admin/settings/reranker-model', { method: 'POST', body: JSON.stringify({ model }) }),
  previewEmbedding:  (model: string, dimension: number) => request<any>('/api/admin/settings/embedding-model/preview', { method: 'POST', body: JSON.stringify({ model, dimension, confirm: false }) }),
  applyEmbedding:    (model: string, dimension: number) => request<any>('/api/admin/settings/embedding-model/apply', { method: 'POST', body: JSON.stringify({ model, dimension, confirm: true }) }),
  queue:             () => request<any>('/api/admin/queue'),
  rebuildIndex:      () => request<any>('/api/admin/rebuild-index', { method: 'POST' }),
};
