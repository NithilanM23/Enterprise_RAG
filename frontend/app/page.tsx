'use client';

import { useState, useEffect, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import { Brain, FileText, MessageSquare, Zap, AlertCircle, ChevronRight } from 'lucide-react';
import { health, sessions as sessionsApi, documents as docsApi } from '@/utils/api';

export default function Dashboard() {
  const router = useRouter();
  const [dbHealth,  setDbHealth]  = useState<any>(null);
  const [kbHealth,  setKbHealth]  = useState<any[]>([]);
  const [docCount,  setDocCount]  = useState(0);
  const [sessionCount, setSessionCount] = useState(0);
  const [recentDocs,   setRecentDocs]   = useState<any[]>([]);
  const [question, setQuestion] = useState('');

  const load = useCallback(async () => {
    try {
      const [h, kb, docs, sess] = await Promise.all([
        health.get(),
        health.knowledgeHealth(),
        docsApi.list(),
        sessionsApi.list(),
      ]);
      setDbHealth(h);
      setKbHealth(kb);
      setDocCount(docs.length);
      setSessionCount(sess.length);
      setRecentDocs(docs.slice(0, 4));
    } catch {}
  }, []);

  useEffect(() => { load(); }, [load]);

  const handleQuickAsk = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim()) return;
    try {
      const s = await sessionsApi.create(question.slice(0, 60));
      router.push(`/chat/${s.id}?q=${encodeURIComponent(question)}`);
    } catch {}
  };

  const greeting = () => {
    const h = new Date().getHours();
    if (h < 12) return 'Good morning';
    if (h < 18) return 'Good afternoon';
    return 'Good evening';
  };

  const StatusDot = ({ ok }: { ok: boolean }) => (
    <div style={{
      width: 6, height: 6, borderRadius: '50%', flexShrink: 0,
      background: ok ? 'var(--success)' : 'var(--error)',
      boxShadow: `0 0 6px ${ok ? 'var(--success)' : 'var(--error)'}`,
    }} />
  );

  return (
    <>
      {/* Topbar */}
      <div className="topbar">
        <Brain size={16} style={{ color: 'var(--primary)' }} />
        <span className="topbar-title">Dashboard</span>
        <div className="topbar-actions" style={{ gap: 16, fontSize: 12, color: 'var(--text-mute)' }}>
          {dbHealth && (
            <>
              <div className="flex items-center gap-2">
                <StatusDot ok={dbHealth.db?.connected} />
                <span>DB</span>
              </div>
              <div className="flex items-center gap-2">
                <StatusDot ok={dbHealth.ollama?.reachable} />
                <span>Ollama</span>
              </div>
            </>
          )}
        </div>
      </div>

      {/* Content */}
      <div className="dashboard">
        {/* Welcome */}
        <div className="dashboard-welcome">
          <h1>{greeting()} 👋</h1>
          <p>Your private knowledge assistant is ready. Ask anything about your documents.</p>
        </div>

        {/* Quick ask */}
        <div className="card">
          <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 12, color: 'var(--text)' }}>
            Ask a question
          </div>
          <form className="quick-ask-form" onSubmit={handleQuickAsk}>
            <input
              type="text"
              placeholder="What does the HR policy say about leave?"
              value={question}
              onChange={e => setQuestion(e.target.value)}
            />
            <button type="submit" className="quick-ask-btn">
              Ask <ChevronRight size={14} />
            </button>
          </form>
        </div>

        {/* Stats row */}
        <div className="dashboard-grid">
          <div className="card">
            <div className="stat-card">
              <div className="stat-value">{docCount}</div>
              <div className="stat-label">Documents ingested</div>
            </div>
          </div>
          <div className="card">
            <div className="stat-card">
              <div className="stat-value">{sessionCount}</div>
              <div className="stat-label">Chat sessions</div>
            </div>
          </div>
        </div>

        {/* Knowledge health */}
        <div className="card">
          <h3 style={{ fontSize: 13, fontWeight: 600, marginBottom: 16, color: 'var(--text)' }}>
            Knowledge Health
          </h3>
          {kbHealth.length === 0 ? (
            <div style={{ fontSize: 13, color: 'var(--text-mute)' }}>
              No documents uploaded yet.{' '}
              <a href="/documents" style={{ color: 'var(--primary-hi)' }}>Upload your first document →</a>
            </div>
          ) : (
            kbHealth.map(item => {
              const pct = item.percent ?? 0;
              const fillCls = pct === 100 ? 'full' : pct >= 50 ? 'partial' : 'low';
              return (
                <div key={item.category} className="health-row">
                  <div className="health-label" title={item.category}>
                    {item.category.replace(/_/g, ' ')}
                    <span style={{ fontSize: 10, color: 'var(--text-mute)', marginLeft: 4 }}>
                      ({item.document_count})
                    </span>
                  </div>
                  <div className="health-bar-wrap">
                    <div
                      className={`health-bar-fill ${fillCls}`}
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                  <div className={`health-pct ${pct < 100 ? 'text-mute' : ''}`}
                    style={{ color: pct < 50 ? 'var(--error)' : pct < 100 ? 'var(--warning)' : 'var(--success)' }}>
                    {pct}%
                  </div>
                  {pct < 100 && (
                    <AlertCircle size={12} style={{ color: 'var(--warning)', flexShrink: 0 }} />
                  )}
                </div>
              );
            })
          )}
          {kbHealth.some(k => k.percent < 100) && (
            <div style={{ marginTop: 12 }}>
              <a href="/documents" className="btn btn-ghost btn-sm">
                Generate Embeddings →
              </a>
            </div>
          )}
        </div>

        {/* Recent documents */}
        {recentDocs.length > 0 && (
          <div className="card">
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 14 }}>
              <h3 style={{ fontSize: 13, fontWeight: 600, color: 'var(--text)' }}>Recent Documents</h3>
              <a href="/documents" style={{ fontSize: 12, color: 'var(--primary-hi)' }}>View all →</a>
            </div>
            {recentDocs.map(doc => (
              <div key={doc.id} style={{
                display: 'flex', alignItems: 'center', gap: 10,
                padding: '9px 0', borderBottom: '1px solid var(--border-sub)',
              }}>
                <FileText size={14} style={{ color: 'var(--text-mute)', flexShrink: 0 }} />
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div className="doc-filename truncate">{doc.filename}</div>
                  <div className="text-mute text-xs" style={{ marginTop: 1 }}>
                    {new Date(doc.upload_time).toLocaleDateString()}
                  </div>
                </div>
                <span className="category-badge">{doc.category?.replace(/_/g, ' ')}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </>
  );
}
