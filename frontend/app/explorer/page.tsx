'use client';

import { useState, useRef } from 'react';
import { BarChart2, Upload, Database, Code, X, RefreshCw } from 'lucide-react';
import { explorer as explorerApi } from '@/utils/api';
import {
  BarChart, Bar, LineChart, Line, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from 'recharts';

interface HistoryItem {
  query: string;
  mode: string;
  result: any;
  result_type: string;
  columns?: string[];
  code?: string;
  image_b64?: string;
  error?: string;
  friendly_error?: string;
  fallback?: boolean;
}

const QUICK_CHIPS = [
  'Describe the data', 'Show first 10 rows', 'How many null values?',
  'Show correlation matrix', 'Value counts for each column', 'Show data types',
];

const CHART_COLORS = ['#1f6feb', '#3fb950', '#d29922', '#f85149', '#58a6ff', '#bc8cff'];

export default function ExplorerPage() {
  const [file,     setFile]     = useState<File | null>(null);
  const [token,    setToken]    = useState<string | null>(null);
  const [schema,   setSchema]   = useState<any>(null);
  const [history,  setHistory]  = useState<HistoryItem[]>([]);
  const [query,    setQuery]    = useState('');
  const [loading,  setLoading]  = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error,    setError]    = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const handleFileChange = async (f: File | null) => {
    if (!f) return;
    setFile(f);
    setUploading(true);
    setError(null);
    try {
      const res = await explorerApi.upload(f);
      if (res.token) {
        setToken(res.token);
        setSchema(res);
        setHistory([]);
      } else {
        setError(res.detail || 'Upload failed');
      }
    } catch (e: any) { setError(e.message); }
    finally { setUploading(false); }
  };

  const handleQuery = async (q: string) => {
    if (!q.trim() || !token || !file || loading) return;
    setLoading(true);
    setQuery('');
    try {
      const res = await explorerApi.query(token, q, file.name);
      setHistory(prev => [...prev, {
        query: q,
        mode: res.mode,
        result: res.result,
        result_type: res.result_type,
        columns: res.columns,
        code: res.code,
        image_b64: res.image_b64,
        error: res.error,
        friendly_error: res.friendly_error,
        fallback: res.fallback,
      }]);
    } catch (e: any) {
      setHistory(prev => [...prev, {
        query: q, mode: 'error', result: null, result_type: 'none',
        error: e.message,
      }]);
    }
    finally { setLoading(false); }
  };

  const renderResult = (item: HistoryItem) => {
    if (item.friendly_error && !item.result) {
      return <div style={{ color: 'var(--error)', fontSize: 13 }}>❌ {item.friendly_error}</div>;
    }
    if (item.error && item.error !== 'BLOCKED' && !item.result) {
      return <div style={{ color: 'var(--error)', fontSize: 13 }}>❌ {item.error}</div>;
    }
    if (item.image_b64) {
      return <img src={`data:image/png;base64,${item.image_b64}`} alt="Chart" style={{ maxWidth: '100%', borderRadius: 8 }} />;
    }
    if (item.result_type === 'dataframe' && Array.isArray(item.result)) {
      const cols = item.columns ?? (item.result[0] ? Object.keys(item.result[0]) : []);
      return (
        <div className="data-table-wrap">
          <table className="data-table">
            <thead><tr>{cols.map(c => <th key={c}>{c}</th>)}</tr></thead>
            <tbody>
              {item.result.slice(0, 100).map((row: any, i: number) => (
                <tr key={i}>{cols.map(c => <td key={c} title={String(row[c] ?? '')}>{String(row[c] ?? '')}</td>)}</tr>
              ))}
            </tbody>
          </table>
          {item.result.length > 100 && (
            <div style={{ fontSize: 11, color: 'var(--text-mute)', padding: '6px 10px' }}>
              Showing 100 of {item.result.length} rows
            </div>
          )}
        </div>
      );
    }
    if (item.result !== null && item.result !== undefined) {
      return <div style={{ fontSize: 13, color: 'var(--text)', fontFamily: 'var(--font-mono)', whiteSpace: 'pre-wrap' }}>{String(item.result)}</div>;
    }
    return null;
  };

  return (
    <>
      {/* Topbar */}
      <div className="topbar">
        <BarChart2 size={15} style={{ color: 'var(--primary)' }} />
        <span className="topbar-title">
          Data Explorer
          {file && <span className="topbar-subtitle" style={{ marginLeft: 10 }}>— {file.name}</span>}
        </span>
        {file && (
          <button className="btn btn-ghost btn-sm" onClick={() => { setFile(null); setToken(null); setSchema(null); setHistory([]); }}>
            <X size={12} /> Close file
          </button>
        )}
      </div>

      {!file ? (
        /* Upload screen */
        <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 40 }}>
          <div style={{ width: '100%', maxWidth: 520 }}>
            <div style={{ textAlign: 'center', marginBottom: 32 }}>
              <BarChart2 size={40} style={{ color: 'var(--primary)', margin: '0 auto 12px' }} />
              <div style={{ fontSize: 20, fontWeight: 600, color: 'var(--text)', marginBottom: 8 }}>
                Data Explorer
              </div>
              <div style={{ fontSize: 13, color: 'var(--text-mute)' }}>
                Upload a CSV or Excel file to explore it with natural language.
                This is a session-only workspace — files are not added to the knowledge base.
              </div>
            </div>

            <div
              className={`upload-zone ${uploading ? 'drag-over' : ''}`}
              onClick={() => fileRef.current?.click()}
              onDragOver={e => e.preventDefault()}
              onDrop={e => { e.preventDefault(); handleFileChange(e.dataTransfer.files[0]); }}
            >
              <input
                ref={fileRef}
                type="file"
                accept=".csv,.xlsx,.xls"
                style={{ display: 'none' }}
                onChange={e => handleFileChange(e.target.files?.[0] ?? null)}
              />
              {uploading ? (
                <><div className="stage-spinner" style={{ margin: '0 auto 12px', width: 24, height: 24 }} />
                  <div className="upload-label">Processing file…</div></>
              ) : (
                <><div className="upload-icon"><Upload size={32} /></div>
                  <div className="upload-label">Drop CSV or Excel here</div>
                  <div className="upload-sub">.csv, .xlsx, .xls — session only, not stored</div></>
              )}
            </div>
            {error && <div style={{ color: 'var(--error)', fontSize: 13, marginTop: 12 }}>❌ {error}</div>}
          </div>
        </div>
      ) : (
        <div className="explorer-layout" style={{ flex: 1 }}>
          {/* Schema panel */}
          <div className="explorer-schema">
            <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--text-mute)', textTransform: 'uppercase', letterSpacing: '.06em', marginBottom: 12 }}>
              Schema
            </div>
            {schema && (
              <>
                <div style={{ fontSize: 12, color: 'var(--text-sec)', marginBottom: 10 }}>
                  {schema.rows?.toLocaleString()} rows × {schema.cols} columns
                </div>
                {(schema.columns ?? []).map((col: string, i: number) => (
                  <div key={i} className="schema-col-row">
                    <Database size={11} style={{ color: 'var(--text-mute)', flexShrink: 0 }} />
                    <span className="schema-col-name">{col}</span>
                  </div>
                ))}
              </>
            )}
          </div>

          {/* Main explorer */}
          <div className="explorer-main">
            {/* Query history */}
            <div className="explorer-results">
              {/* Quick chips */}
              <div className="chip-row">
                {QUICK_CHIPS.map(chip => (
                  <button key={chip} className="chip" onClick={() => handleQuery(chip)} disabled={loading}>
                    {chip}
                  </button>
                ))}
              </div>

              {history.length === 0 && (
                <div className="empty-state">
                  <div className="empty-icon"><BarChart2 size={36} /></div>
                  <div className="empty-title">Start exploring</div>
                  <div className="empty-sub">Use the quick chips above or type any question about your data below.</div>
                </div>
              )}

              {history.map((item, i) => (
                <div key={i} style={{ marginBottom: 24 }}>
                  {/* Query */}
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
                    <div style={{
                      background: 'var(--primary)', color: '#fff',
                      padding: '5px 12px', borderRadius: 'var(--r-md)',
                      fontSize: 13,
                    }}>
                      {item.query}
                    </div>
                    <span style={{ fontSize: 11, color: 'var(--text-mute)', background: 'var(--surface-hi2)', padding: '2px 8px', borderRadius: 10, border: '1px solid var(--border-sub)' }}>
                      {item.mode}
                    </span>
                    {item.fallback && (
                      <span style={{ fontSize: 11, color: 'var(--warning)' }}>⟳ auto fallback</span>
                    )}
                  </div>

                  {/* Result */}
                  <div className="card" style={{ padding: 14 }}>
                    {renderResult(item)}

                    {/* Code disclosure */}
                    {item.code && item.mode !== 'statistical' && (
                      <details style={{ marginTop: 10 }}>
                        <summary style={{ fontSize: 11, color: 'var(--text-mute)', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 6 }}>
                          <Code size={11} /> Generated code
                        </summary>
                        <pre style={{ marginTop: 8, padding: '10px 12px', background: 'var(--surface-hi2)', borderRadius: 'var(--r-sm)', fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--text-sec)', overflow: 'auto' }}>
                          {item.code}
                        </pre>
                      </details>
                    )}
                  </div>
                </div>
              ))}

              {loading && (
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '12px 0', color: 'var(--text-mute)', fontSize: 13 }}>
                  <div className="stage-spinner" />
                  Analysing…
                </div>
              )}
            </div>

            {/* Input */}
            <div className="chat-input-area">
              <div className="chat-input-wrap">
                <textarea
                  className="chat-input"
                  rows={1}
                  placeholder="Ask anything about the data… (Enter to run)"
                  value={query}
                  onChange={e => setQuery(e.target.value)}
                  onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleQuery(query); } }}
                  disabled={loading}
                />
                <button
                  className="chat-send-btn"
                  onClick={() => handleQuery(query)}
                  disabled={!query.trim() || loading}
                >
                  <BarChart2 size={15} />
                </button>
              </div>
              {history.length > 0 && (
                <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: 6 }}>
                  <button className="btn btn-ghost btn-sm" onClick={() => setHistory([])}>
                    <RefreshCw size={11} /> Clear history
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </>
  );
}
