'use client';

import { useState, useEffect, useCallback } from 'react';
import { Settings, RefreshCw, AlertTriangle, CheckCircle, Tag, Plus, Trash2, Zap } from 'lucide-react';
import { admin as adminApi, categories as catsApi } from '@/utils/api';
import { useAuth } from '../context/AuthContext';
import { useRouter } from 'next/navigation';

const SAFE_SETTINGS = [
  { key: 'chunk_size',                   label: 'Chunk Size',                   type: 'int',   hint: 'Characters per chunk. Re-ingest documents after changing.' },
  { key: 'chunk_overlap',               label: 'Chunk Overlap',               type: 'int',   hint: 'Overlap between adjacent chunks.' },
  { key: 'top_k',                       label: 'Top K Results',               type: 'int',   hint: 'Final chunks sent to LLM per query (3–10).' },
  { key: 'semantic_k',                  label: 'Semantic K',                  type: 'int',   hint: 'Candidates from vector search.' },
  { key: 'bm25_k',                      label: 'BM25 K',                      type: 'int',   hint: 'Candidates from keyword search.' },
  { key: 'mmr_pool',                    label: 'MMR Pool',                    type: 'int',   hint: 'Candidates entering MMR diversification.' },
  { key: 'mmr_lambda',                  label: 'MMR Lambda',                  type: 'float', hint: '0 = diverse, 1 = relevant. Default 0.85.' },
  { key: 'history_window',              label: 'History Window',              type: 'int',   hint: 'Previous messages injected into each prompt.' },
  { key: 'num_predict',                 label: 'Max Tokens (num_predict)',     type: 'int',   hint: 'Maximum LLM output tokens per response.' },
  { key: 'temperature',                 label: 'Temperature',                 type: 'float', hint: '0 = deterministic, 1 = creative. Default 0.1.' },
  { key: 'routing_confidence_threshold', label: 'Routing Threshold',          type: 'float', hint: 'Min score to hard-scope search to a category.' },
];

export default function AdminPage() {
  const { username } = useAuth();
  const router = useRouter();
  const [settings,      setSettings]      = useState<any>({});
  const [ollamaModels,  setOllamaModels]  = useState<string[]>([]);
  const [localSettings, setLocalSettings] = useState<any>({});
  const [cats,          setCats]          = useState<any[]>([]);
  const [newCatLabel,   setNewCatLabel]   = useState('');
  const [newCatKw,      setNewCatKw]      = useState('');
  const [queue,         setQueue]         = useState<any>(null);
  const [toast,         setToast]         = useState<{ msg: string; type?: string } | null>(null);
  const [saving,        setSaving]        = useState(false);
  const [llmSwapping,   setLlmSwapping]   = useState(false);
  const [embedPreview,  setEmbedPreview]  = useState<any>(null);
  const [newEmbedModel, setNewEmbedModel] = useState('');
  const [newEmbedDim,   setNewEmbedDim]   = useState('');

  const showToast = (msg: string, type = 'success') => {
    setToast({ msg, type });
    setTimeout(() => setToast(null), 3500);
  };

  const load = useCallback(async () => {
    try {
      const [s, c, q] = await Promise.all([
        adminApi.getSettings(),
        catsApi.list(),
        adminApi.queue(),
      ]);
      setSettings(s);
      setLocalSettings(s.settings ?? {});
      setOllamaModels(s.available_ollama_models ?? []);
      setCats(c);
      setQueue(q);
    } catch {}
  }, []);

  useEffect(() => { load(); }, [load]);

  const handleSaveSettings = async () => {
    setSaving(true);
    try {
      const diff: any = {};
      const current = settings.settings ?? {};
      for (const { key } of SAFE_SETTINGS) {
        const v = localSettings[key];
        if (v !== undefined && String(v) !== String(current[key])) diff[key] = v;
      }
      if (Object.keys(diff).length === 0) { showToast('No changes to save.', 'info'); return; }
      await adminApi.updateSettings(diff);
      showToast(`Saved ${Object.keys(diff).length} setting(s). Applied immediately.`);
      load();
    } catch (e: any) { showToast(e.message, 'error'); }
    finally { setSaving(false); }
  };

  const handleLLMSwap = async () => {
    const model = localSettings.llm_model;
    if (!model) return;
    setLlmSwapping(true);
    try {
      await adminApi.swapLLM(model);
      showToast(`LLM model swapped to "${model}". Applies on next question.`);
      load();
    } catch (e: any) { showToast(e.message, 'error'); }
    finally { setLlmSwapping(false); }
  };

  const handleRerankerSwap = async () => {
    const model = localSettings.reranker_model;
    if (!model) return;
    try {
      const res = await adminApi.swapReranker(model);
      showToast(
        res.already_cached
          ? `Reranker swapped to "${model}" (already cached).`
          : `Reranker swapped to "${model}". First use will download the model.`,
        'info'
      );
      load();
    } catch (e: any) { showToast(e.message, 'error'); }
  };

  const handleEmbedPreview = async () => {
    if (!newEmbedModel || !newEmbedDim) return;
    try {
      const res = await adminApi.previewEmbedding(newEmbedModel, Number(newEmbedDim));
      setEmbedPreview(res);
    } catch (e: any) { showToast(e.message, 'error'); }
  };

  const handleEmbedApply = async () => {
    if (!embedPreview) return;
    if (!confirm(`This will NULL ${embedPreview.chunks_to_reembed} embeddings. Continue?`)) return;
    try {
      await adminApi.applyEmbedding(newEmbedModel, Number(newEmbedDim));
      showToast('Embedding model applied. Run "Generate Embeddings" to rebuild.', 'info');
      setEmbedPreview(null);
      load();
    } catch (e: any) { showToast(e.message, 'error'); }
  };

  const handleRebuildIndex = async () => {
    try {
      await adminApi.rebuildIndex();
      showToast('BM25 index rebuild started.');
    } catch (e: any) { showToast(e.message, 'error'); }
  };

  const handleCreateCat = async () => {
    if (!newCatLabel.trim()) return;
    try {
      const kw = newCatKw.split(',').map(s => s.trim()).filter(Boolean);
      await catsApi.create(newCatLabel.trim(), kw);
      showToast(`Category "${newCatLabel}" created.`);
      setNewCatLabel('');
      setNewCatKw('');
      load();
    } catch (e: any) { showToast(e.message, 'error'); }
  };

  const handleDeleteCat = async (name: string) => {
    if (!confirm(`Delete category "${name}"? Documents using it will be reassigned to "general".`)) return;
    try {
      await catsApi.delete(name);
      showToast(`Category "${name}" deleted.`);
      load();
    } catch (e: any) { showToast(e.message, 'error'); }
  };

  const Field = ({ k, label, type, hint }: { k: string; label: string; type: string; hint: string }) => (
    <div className="settings-field">
      <label>{label}</label>
      <input
        type="number"
        step={type === 'float' ? 0.05 : 1}
        min={0}
        value={localSettings[k] ?? ''}
        onChange={e => setLocalSettings((p: any) => ({ ...p, [k]: type === 'float' ? parseFloat(e.target.value) : parseInt(e.target.value) }))}
        title={hint}
      />
      <div style={{ fontSize: 11, color: 'var(--text-mute)', marginTop: 4 }}>{hint}</div>
    </div>
  );

  useEffect(() => {
    if (username && username !== 'admin') {
      router.push('/');
    }
  }, [username, router]);

  if (username !== 'admin') {
    return null;
  }

  return (
    <>
      {/* Topbar */}
      <div className="topbar">
        <Settings size={15} style={{ color: 'var(--primary)' }} />
        <span className="topbar-title">Admin Settings</span>
        {queue && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, color: 'var(--text-mute)', marginLeft: 'auto' }}>
            <Zap size={12} />
            Ollama: {queue.active} active, {queue.queued} queued
            {queue.queued > 0 && <span style={{ color: 'var(--warning)' }}>~{queue.estimated_wait_seconds}s wait</span>}
          </div>
        )}
      </div>

      <div className="page-content">

        {/* ── Retrieval settings ─────────────────────────────── */}
        <div className="card">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 20 }}>
            <h2 style={{ fontSize: 14, fontWeight: 600, color: 'var(--text)' }}>Retrieval & Generation</h2>
            <div style={{ display: 'flex', gap: 8 }}>
              <button className="btn btn-ghost btn-sm" onClick={load}><RefreshCw size={11} /> Refresh</button>
              <button className="btn btn-primary btn-sm" onClick={handleSaveSettings} disabled={saving}>
                {saving ? <><div className="stage-spinner" />Saving…</> : '💾 Save Changes'}
              </button>
            </div>
          </div>

          <div className="settings-grid">
            {SAFE_SETTINGS.map(({ key, ...rest }) => <Field key={key} k={key} {...rest} />)}
          </div>
        </div>

        {/* ── LLM Model ─────────────────────────────────────── */}
        <div className="card">
          <h2 style={{ fontSize: 14, fontWeight: 600, color: 'var(--text)', marginBottom: 4 }}>LLM Model</h2>
          <p style={{ fontSize: 12, color: 'var(--text-mute)', marginBottom: 16 }}>
            Current: <strong style={{ color: 'var(--text)' }}>{settings.settings?.llm_model}</strong>
            {settings.llm_model_available
              ? <CheckCircle size={11} style={{ marginLeft: 6, color: 'var(--success)', verticalAlign: 'middle' }} />
              : <AlertTriangle size={11} style={{ marginLeft: 6, color: 'var(--warning)', verticalAlign: 'middle' }} />}
          </p>
          <div style={{ display: 'flex', gap: 10 }}>
            <select
              value={localSettings.llm_model ?? ''}
              onChange={e => setLocalSettings((p: any) => ({ ...p, llm_model: e.target.value }))}
              style={{ flex: 1 }}
            >
              {ollamaModels.map(m => <option key={m} value={m}>{m}</option>)}
            </select>
            <button className="btn btn-primary" onClick={handleLLMSwap} disabled={llmSwapping}>
              {llmSwapping ? <div className="stage-spinner" /> : 'Swap'}
            </button>
          </div>
          <p style={{ fontSize: 11, color: 'var(--text-mute)', marginTop: 8 }}>
            Safe — applies on the next question. No data migration needed.
          </p>
        </div>

        {/* ── Reranker Model ────────────────────────────────── */}
        <div className="card">
          <h2 style={{ fontSize: 14, fontWeight: 600, color: 'var(--text)', marginBottom: 4 }}>Reranker Model</h2>
          <p style={{ fontSize: 12, color: 'var(--text-mute)', marginBottom: 16 }}>
            Current: <strong style={{ color: 'var(--text)' }}>{settings.settings?.reranker_model}</strong>
          </p>
          <div style={{ display: 'flex', gap: 10 }}>
            <input
              type="text"
              placeholder="cross-encoder/ms-marco-MiniLM-L-6-v2"
              value={localSettings.reranker_model ?? ''}
              onChange={e => setLocalSettings((p: any) => ({ ...p, reranker_model: e.target.value }))}
            />
            <button className="btn btn-primary" onClick={handleRerankerSwap}>Swap</button>
          </div>
          <p style={{ fontSize: 11, color: 'var(--text-mute)', marginTop: 8 }}>
            Any HuggingFace cross-encoder model. First use after swap downloads the model (~80MB typical).
          </p>
        </div>

        {/* ── Embedding Model ───────────────────────────────── */}
        <div className="card" style={{ borderColor: embedPreview ? 'var(--warning)' : undefined }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
            <h2 style={{ fontSize: 14, fontWeight: 600, color: 'var(--text)' }}>Embedding Model</h2>
            <span style={{ fontSize: 11, padding: '2px 8px', borderRadius: 10, background: '#f8514922', color: 'var(--error)', border: '1px solid var(--error)' }}>
              Destructive
            </span>
          </div>
          <p style={{ fontSize: 12, color: 'var(--text-mute)', marginBottom: 16 }}>
            Current: <strong style={{ color: 'var(--text)' }}>{settings.settings?.embedding_model}</strong>
            {' '}(dim={settings.settings?.embedding_dimension})
          </p>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr auto auto', gap: 10, marginBottom: 10 }}>
            <select
              value={newEmbedModel}
              onChange={e => setNewEmbedModel(e.target.value)}
            >
              <option value="">-- select model --</option>
              {ollamaModels.map(m => <option key={m} value={m}>{m}</option>)}
            </select>
            <input
              type="number"
              placeholder="Dimension"
              value={newEmbedDim}
              onChange={e => setNewEmbedDim(e.target.value)}
              style={{ width: 100 }}
            />
            <button className="btn btn-ghost" onClick={handleEmbedPreview} disabled={!newEmbedModel || !newEmbedDim}>
              Preview
            </button>
          </div>

          {embedPreview && (
            <div style={{ background: 'var(--surface-hi2)', border: '1px solid var(--warning)', borderRadius: 'var(--r-md)', padding: '14px 16px', marginBottom: 12 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10, color: 'var(--warning)', fontWeight: 600, fontSize: 13 }}>
                <AlertTriangle size={14} /> Warning — Destructive Operation
              </div>
              <div style={{ fontSize: 12, color: 'var(--text-sec)', lineHeight: 1.7 }}>
                {embedPreview.warning}
              </div>
              <div style={{ display: 'flex', gap: 10, marginTop: 14 }}>
                <button className="btn btn-danger btn-sm" onClick={handleEmbedApply}>
                  Apply — Null {embedPreview.chunks_to_reembed} embeddings
                </button>
                <button className="btn btn-ghost btn-sm" onClick={() => setEmbedPreview(null)}>Cancel</button>
              </div>
            </div>
          )}

          <p style={{ fontSize: 11, color: 'var(--text-mute)' }}>
            Changing the embedding model nulls all existing embeddings (incompatible vector spaces).
            You must re-run Generate Embeddings after applying.
          </p>
        </div>

        {/* ── Categories ────────────────────────────────────── */}
        <div className="card">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
            <h2 style={{ fontSize: 14, fontWeight: 600, color: 'var(--text)' }}>
              <Tag size={14} style={{ marginRight: 6, verticalAlign: 'middle' }} />
              Document Categories
            </h2>
          </div>

          {/* Create category */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr auto', gap: 10, marginBottom: 20, padding: '14px 16px', background: 'var(--surface-hi2)', borderRadius: 'var(--r-md)' }}>
            <div>
              <label style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-mute)', display: 'block', marginBottom: 5 }}>Category Name</label>
              <input type="text" placeholder="e.g. Quality Control" value={newCatLabel} onChange={e => setNewCatLabel(e.target.value)} />
            </div>
            <div>
              <label style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-mute)', display: 'block', marginBottom: 5 }}>Keywords (comma-separated)</label>
              <input type="text" placeholder="quality, inspection, defect, tolerance" value={newCatKw} onChange={e => setNewCatKw(e.target.value)} />
            </div>
            <div style={{ display: 'flex', alignItems: 'flex-end' }}>
              <button className="btn btn-primary" onClick={handleCreateCat}>
                <Plus size={13} /> Create
              </button>
            </div>
          </div>
          <p style={{ fontSize: 11, color: 'var(--text-mute)', marginTop: -12, marginBottom: 16 }}>
            Keywords improve automatic routing. Leave empty for a new category — keywords can be added later.
          </p>

          {/* Category table */}
          <table className="doc-table">
            <thead>
              <tr>
                <th>Name</th>
                <th>Label</th>
                <th>Keywords</th>
                <th>Docs</th>
                <th>Type</th>
                <th style={{ textAlign: 'right' }}>Actions</th>
              </tr>
            </thead>
            <tbody>
              {cats.map(cat => (
                <tr key={cat.id}>
                  <td><code style={{ fontSize: 12, fontFamily: 'var(--font-mono)' }}>{cat.name}</code></td>
                  <td style={{ color: 'var(--text)' }}>{cat.label}</td>
                  <td style={{ maxWidth: 200 }}>
                    <div className="truncate" style={{ fontSize: 11, color: 'var(--text-mute)' }}>
                      {cat.keywords?.length ? cat.keywords.slice(0, 5).join(', ') + (cat.keywords.length > 5 ? `… +${cat.keywords.length - 5}` : '') : '—'}
                    </div>
                  </td>
                  <td>{cat.document_count ?? 0}</td>
                  <td>
                    <span style={{ fontSize: 11, padding: '2px 7px', borderRadius: 10, background: cat.is_custom ? 'var(--primary-dim)' : 'var(--surface-hi2)', color: cat.is_custom ? 'var(--primary-hi)' : 'var(--text-mute)' }}>
                      {cat.is_custom ? 'Custom' : 'Built-in'}
                    </span>
                  </td>
                  <td style={{ textAlign: 'right' }}>
                    {cat.is_custom && (
                      <button className="btn btn-danger btn-sm" onClick={() => handleDeleteCat(cat.name)}>
                        <Trash2 size={11} />
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* ── Admin ops ────────────────────────────────────── */}
        <div className="card">
          <h2 style={{ fontSize: 14, fontWeight: 600, color: 'var(--text)', marginBottom: 16 }}>Server Operations</h2>
          <div style={{ display: 'flex', gap: 10 }}>
            <button className="btn btn-ghost" onClick={handleRebuildIndex}>
              <RefreshCw size={13} /> Rebuild BM25 Index
            </button>
            <button className="btn btn-ghost" onClick={load}>
              <RefreshCw size={13} /> Refresh Status
            </button>
          </div>
        </div>
      </div>

      {toast && (
        <div className="toast-container">
          <div className={`toast ${toast.type ?? 'success'}`}>{toast.msg}</div>
        </div>
      )}
    </>
  );
}
