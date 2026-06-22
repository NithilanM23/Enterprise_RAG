'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import {
  FileText, Upload, Trash2, RefreshCw, Plus, X,
  CheckCircle, AlertCircle, Loader,
} from 'lucide-react';
import { documents as docsApi, categories as catsApi } from '@/utils/api';

interface IngestionCard {
  filename: string;
  doc_id: number;
  stage: string;
  percent: number;
  error: string | null;
}

export default function DocumentsPage() {
  const [docs,       setDocs]       = useState<any[]>([]);
  const [cats,       setCats]       = useState<any[]>([]);
  const [ingesting,  setIngesting]  = useState<IngestionCard[]>([]);
  const [dragOver,   setDragOver]   = useState(false);
  const [category,   setCategory]   = useState('general');
  const [newCatLabel, setNewCatLabel] = useState('');
  const [toast,      setToast]      = useState<{ msg: string; type: string } | null>(null);
  const [embedding,  setEmbedding]  = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const pollRefs     = useRef<Record<number, ReturnType<typeof setInterval>>>({});

  const showToast = (msg: string, type = 'success') => {
    setToast({ msg, type });
    setTimeout(() => setToast(null), 3000);
  };

  const load = useCallback(async () => {
    try {
      const [d, c] = await Promise.all([docsApi.list(), catsApi.list()]);
      setDocs(d);
      setCats(c);
    } catch {}
  }, []);

  useEffect(() => { load(); }, [load]);

  // Poll ingestion progress for each uploading doc
  const startPolling = (docId: number, filename: string) => {
    if (pollRefs.current[docId]) return;
    pollRefs.current[docId] = setInterval(async () => {
      try {
        const status = await docsApi.status(docId);
        setIngesting(prev =>
          prev.map(c => c.doc_id === docId
            ? { ...c, stage: status.stage, percent: status.percent_complete, error: status.error }
            : c
          )
        );
        if (status.stage === 'ready' || status.stage === 'failed') {
          clearInterval(pollRefs.current[docId]);
          delete pollRefs.current[docId];
          if (status.stage === 'ready') {
            setTimeout(() => {
              setIngesting(prev => prev.filter(c => c.doc_id !== docId));
              load();
            }, 2000);
          }
        }
      } catch {}
    }, 3000);
  };

  const handleUpload = async (files: FileList | null) => {
    if (!files || files.length === 0) return;
    const resolvedCat = newCatLabel.trim() ? undefined : category;

    for (const file of Array.from(files)) {
      try {
        const result = await docsApi.upload(file, resolvedCat || category, newCatLabel.trim() || undefined);
        if (result.doc_id || result.message) {
          // Find the doc_id — it may be in result directly or need a list refresh
          const docId = result.doc_id ?? (await docsApi.list()).find((d: any) => d.filename === file.name)?.id;
          if (docId) {
            const card: IngestionCard = { filename: file.name, doc_id: docId, stage: 'chunking', percent: 0, error: null };
            setIngesting(prev => [...prev, card]);
            startPolling(docId, file.name);
          }
          showToast(`Uploading ${file.name}…`, 'info');
        } else if (result.detail) {
          showToast(result.detail, 'error');
        }
      } catch (e: any) { showToast(e.message, 'error'); }
    }
    if (newCatLabel.trim()) setNewCatLabel('');
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    handleUpload(e.dataTransfer.files);
  };

  const handleDelete = async (id: number, filename: string) => {
    if (!confirm(`Delete "${filename}"? This removes all its chunks and embeddings.`)) return;
    try {
      await docsApi.delete(id);
      setDocs(prev => prev.filter(d => d.id !== id));
      showToast(`"${filename}" deleted.`);
    } catch (e: any) { showToast(e.message, 'error'); }
  };

  const handleEmbed = async () => {
    setEmbedding(true);
    try {
      await docsApi.embed();
      showToast('Embedding started in background. Refresh in a moment.', 'info');
      setTimeout(load, 5000);
    } catch (e: any) { showToast(e.message, 'error'); }
    finally { setEmbedding(false); }
  };

  const StageChip = ({ stage, pct }: { stage: string; pct: number }) => {
    const map: Record<string, [string, JSX.Element]> = {
      chunking:  ['stage-chunking',  <><div className="stage-spinner" />Chunking…</>],
      embedding: ['stage-embedding', <><div className="stage-spinner" />Embedding {pct}%</>],
      ready:     ['stage-ready',     <><CheckCircle size={10} />Ready</>],
      failed:    ['stage-failed',    <><AlertCircle size={10} />Failed</>],
    };
    const [cls, content] = map[stage] ?? ['stage-chunking', <><div className="stage-spinner" />Processing…</>];
    return <span className={`stage-badge ${cls}`}>{content}</span>;
  };

  const catOptions = [{ name: 'general', label: 'General' }, ...cats.filter(c => c.name !== 'general')];

  return (
    <>
      {/* Topbar */}
      <div className="topbar">
        <FileText size={15} style={{ color: 'var(--primary)' }} />
        <span className="topbar-title">Documents</span>
        <div className="topbar-actions">
          <button className="btn btn-ghost btn-sm" onClick={load}>
            <RefreshCw size={12} /> Refresh
          </button>
          <button
            className="btn btn-primary btn-sm"
            onClick={handleEmbed}
            disabled={embedding}
          >
            {embedding ? <><div className="stage-spinner" />Embedding…</> : <><RefreshCw size={12} />Generate Embeddings</>}
          </button>
        </div>
      </div>

      <div className="page-content">
        {/* Upload section */}
        <div className="card">
          <div className="page-header" style={{ marginBottom: 16 }}>
            <div className="page-title" style={{ fontSize: 14 }}>Upload Documents</div>
          </div>

          {/* Category selectors */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 16 }}>
            <div>
              <label style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-sec)', display: 'block', marginBottom: 6 }}>
                Category
              </label>
              <select value={category} onChange={e => setCategory(e.target.value)}>
                {catOptions.map(c => (
                  <option key={c.name} value={c.name}>{c.label}</option>
                ))}
              </select>
            </div>
            <div>
              <label style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-sec)', display: 'block', marginBottom: 6 }}>
                Or create new category
              </label>
              <input
                type="text"
                placeholder="e.g. Quality Control"
                value={newCatLabel}
                onChange={e => setNewCatLabel(e.target.value)}
              />
            </div>
          </div>

          {/* Drop zone */}
          <div
            className={`upload-zone ${dragOver ? 'drag-over' : ''}`}
            onDragOver={e => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept=".pdf,.txt,.docx,.pptx,.ppt,.xlsx,.xls"
              onChange={e => handleUpload(e.target.files)}
              style={{ display: 'none' }}
            />
            <div className="upload-icon"><Upload size={32} /></div>
            <div className="upload-label">Drop files here or click to browse</div>
            <div className="upload-sub">PDF, TXT, DOCX, PPTX, XLSX — up to 50MB each</div>
          </div>
        </div>

        {/* Active ingestion cards */}
        {ingesting.length > 0 && (
          <div className="card">
            <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 12, color: 'var(--text)' }}>
              Processing
            </div>
            {ingesting.map(card => (
              <div key={card.doc_id} style={{
                display: 'flex', alignItems: 'center', gap: 12,
                padding: '10px 0', borderBottom: '1px solid var(--border-sub)',
              }}>
                <FileText size={14} style={{ color: 'var(--text-mute)', flexShrink: 0 }} />
                <span className="truncate" style={{ flex: 1, fontSize: 13 }}>{card.filename}</span>
                <StageChip stage={card.stage} pct={Math.round(card.percent)} />
                {card.stage === 'embedding' && (
                  <div style={{ width: 80, height: 4, background: 'var(--surface-hi2)', borderRadius: 2, overflow: 'hidden' }}>
                    <div style={{ height: '100%', width: `${card.percent}%`, background: 'var(--primary)', transition: 'width .3s' }} />
                  </div>
                )}
                {card.error && <span style={{ fontSize: 11, color: 'var(--error)' }}>{card.error}</span>}
              </div>
            ))}
          </div>
        )}

        {/* Document list */}
        <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
          <div style={{ padding: '16px 20px', borderBottom: '1px solid var(--border-sub)', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <span style={{ fontSize: 14, fontWeight: 600, color: 'var(--text)' }}>
              Knowledge Base
            </span>
            <span style={{ fontSize: 12, color: 'var(--text-mute)' }}>{docs.length} document{docs.length !== 1 ? 's' : ''}</span>
          </div>

          {docs.length === 0 ? (
            <div className="empty-state">
              <div className="empty-icon"><FileText size={36} /></div>
              <div className="empty-title">No documents yet</div>
              <div className="empty-sub">Upload your first document above to get started.</div>
            </div>
          ) : (
            <table className="doc-table">
              <thead>
                <tr>
                  <th>Document</th>
                  <th>Category</th>
                  <th>Uploaded</th>
                  <th style={{ textAlign: 'right' }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {docs.map(doc => (
                  <tr key={doc.id}>
                    <td>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <FileText size={13} style={{ color: 'var(--text-mute)', flexShrink: 0 }} />
                        <div>
                          <div className="doc-filename">{doc.filename}</div>
                          <div className="text-xs text-mute" style={{ marginTop: 1 }}>ID: {doc.id}</div>
                        </div>
                      </div>
                    </td>
                    <td><span className="category-badge">{doc.category?.replace(/_/g, ' ')}</span></td>
                    <td style={{ color: 'var(--text-mute)', fontSize: 12 }}>
                      {new Date(doc.upload_time).toLocaleDateString()}
                    </td>
                    <td style={{ textAlign: 'right' }}>
                      <button
                        className="btn btn-danger btn-sm"
                        onClick={() => handleDelete(doc.id, doc.filename)}
                      >
                        <Trash2 size={11} />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>

      {/* Toast */}
      {toast && (
        <div className="toast-container">
          <div className={`toast ${toast.type}`}>{toast.msg}</div>
        </div>
      )}
    </>
  );
}
