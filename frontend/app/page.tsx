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

  const StatusDot = ({ ok, color = 'var(--success)' }: { ok: boolean, color?: string }) => (
    <div style={{
      width: 6, height: 6, borderRadius: '50%', flexShrink: 0,
      background: ok ? color : 'var(--error)',
      boxShadow: `0 0 6px ${ok ? color : 'var(--error)'}`,
    }} />
  );

  return (
    <div className="dashboard-container">
      {/* Floating Status Badges */}
      <div className="dashboard-status">
        {dbHealth && (
          <>
            <div className={`status-pill ${dbHealth.db?.connected ? 'ok' : 'error'}`}>
              <StatusDot ok={dbHealth.db?.connected} /> DB
            </div>
            <div className={`status-pill ${dbHealth.ollama?.reachable ? 'ok' : 'error'}`}>
              <StatusDot ok={dbHealth.ollama?.reachable} /> Ollama
            </div>
          </>
        )}
      </div>

      <div className="dashboard-content">
        {/* Hero Section */}
        <div className="dashboard-hero">
          <h1 className="hero-greeting">{greeting()}</h1>
          <p className="hero-subtitle">Your private knowledge assistant is ready.</p>
        </div>

        {/* Floating Input Box */}
        <div className="hero-search-wrap">
          <form className="hero-search-form" onSubmit={handleQuickAsk}>
            <input
              type="text"
              className="hero-search-input"
              placeholder="Ask anything about your documents..."
              value={question}
              onChange={e => setQuestion(e.target.value)}
            />
            <button type="submit" className="hero-search-btn" disabled={!question.trim()}>
              <ChevronRight size={20} />
            </button>
          </form>
        </div>

        <div className="dashboard-grid">
          {/* Knowledge Health */}
          <div className="dashboard-column">
            <div className="dashboard-panel" style={{ height: '100%' }}>
              <h3 className="panel-title">Knowledge Health</h3>
              {kbHealth.length === 0 ? (
                <div className="panel-empty">
                  No documents uploaded yet.{' '}
                  <a href="/documents">Upload your first document →</a>
                </div>
              ) : (
                <div className="health-list">
                  {kbHealth.map(item => {
                    const pct = item.percent ?? 0;
                    return (
                      <div key={item.category} className="health-item">
                        <div className="health-header">
                          <span className="health-category">{item.category.replace(/_/g, ' ')}</span>
                          <span className="health-pct" style={{ color: pct < 50 ? 'var(--error)' : pct < 100 ? 'var(--warning)' : 'var(--success)' }}>
                            {pct}%
                          </span>
                        </div>
                        <div className="health-track">
                          <div
                            className="health-fill"
                            style={{ 
                              width: `${pct}%`,
                              background: pct < 50 ? 'var(--error)' : pct < 100 ? 'var(--warning)' : 'var(--success)'
                            }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </div>

          {/* Recent Documents */}
          <div className="dashboard-column">
            <div className="dashboard-panel" style={{ height: '100%' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
                <h3 className="panel-title" style={{ marginBottom: 0 }}>Recent Documents</h3>
                <a href="/documents" className="view-all-link">View all</a>
              </div>
              
              {recentDocs.length === 0 ? (
                <div className="panel-empty">No recent documents</div>
              ) : (
                <div className="recent-docs-list">
                  {recentDocs.map(doc => (
                    <div key={doc.id} className="recent-doc-item">
                      <div className="recent-doc-icon">
                        <FileText size={16} />
                      </div>
                      <div className="recent-doc-info">
                        <div className="recent-doc-name">{doc.filename}</div>
                        <div className="recent-doc-date">{new Date(doc.upload_time).toLocaleDateString()}</div>
                      </div>
                      {doc.category && (
                        <div className="recent-doc-badge">{doc.category.replace(/_/g, ' ')}</div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
