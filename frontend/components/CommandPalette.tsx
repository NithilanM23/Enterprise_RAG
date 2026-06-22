'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { Search, MessagesSquare, FileText, BarChart2, Bookmark, Settings, X } from 'lucide-react';
import { sessions as sessionsApi } from '@/utils/api';

interface Props {
  onClose: () => void;
  onNavigate: (path: string) => void;
}

const NAV_ITEMS = [
  { label: 'Dashboard',     path: '/',          icon: <Search size={14} /> },
  { label: 'Documents',     path: '/documents', icon: <FileText size={14} /> },
  { label: 'Data Explorer', path: '/explorer',  icon: <BarChart2 size={14} /> },
  { label: 'Saved Answers', path: '/saved',     icon: <Bookmark size={14} /> },
  { label: 'Admin',         path: '/admin',     icon: <Settings size={14} /> },
];

export default function CommandPalette({ onClose, onNavigate }: Props) {
  const [query,    setQuery]   = useState('');
  const [results,  setResults] = useState<any[]>([]);
  const [selected, setSelected] = useState(0);
  const [loading,  setLoading] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => { inputRef.current?.focus(); }, []);

  const search = useCallback(async (q: string) => {
    if (!q.trim()) { setResults([]); return; }
    setLoading(true);
    try {
      const sessionResults = await sessionsApi.search(q);
      setResults(sessionResults);
      setSelected(0);
    } catch { setResults([]); }
    finally { setLoading(false); }
  }, []);

  useEffect(() => {
    const t = setTimeout(() => search(query), 200);
    return () => clearTimeout(t);
  }, [query, search]);

  // Filtered nav items
  const navItems = NAV_ITEMS.filter(n =>
    !query || n.label.toLowerCase().includes(query.toLowerCase())
  );

  const allItems = [
    ...navItems.map(n => ({ type: 'nav', ...n })),
    ...results.map(r => ({ type: 'session', label: r.title, path: `/chat/${r.session_id}`, snippet: r.snippet, icon: <MessagesSquare size={14} /> })),
  ];

  const handleKey = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') { e.preventDefault(); setSelected(s => Math.min(s + 1, allItems.length - 1)); }
    if (e.key === 'ArrowUp')   { e.preventDefault(); setSelected(s => Math.max(s - 1, 0)); }
    if (e.key === 'Enter' && allItems[selected]) { onNavigate(allItems[selected].path); }
    if (e.key === 'Escape') { onClose(); }
  };

  return (
    <div className="palette-overlay" onClick={onClose}>
      <div className="palette-box" onClick={e => e.stopPropagation()}>
        {/* Input */}
        <div className="palette-input-row">
          <Search size={16} style={{ color: 'var(--text-mute)', flexShrink: 0 }} />
          <input
            ref={inputRef}
            className="palette-input"
            placeholder="Search sessions, documents, pages..."
            value={query}
            onChange={e => setQuery(e.target.value)}
            onKeyDown={handleKey}
          />
          {loading && <div className="stage-spinner" />}
          <button onClick={onClose} style={{ color: 'var(--text-mute)', padding: 4 }}>
            <X size={14} />
          </button>
        </div>

        {/* Results */}
        <div className="palette-results">
          {!query && (
            <div style={{ padding: '6px 12px 4px', fontSize: 11, color: 'var(--text-mute)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '.04em' }}>
              Pages
            </div>
          )}

          {allItems.length === 0 && query && (
            <div className="palette-empty">No results for &ldquo;{query}&rdquo;</div>
          )}

          {allItems.map((item, i) => (
            <div key={`${item.type}-${i}`}>
              {item.type === 'session' && i > 0 && allItems[i-1].type !== 'session' && (
                <div style={{ padding: '8px 12px 4px', fontSize: 11, color: 'var(--text-mute)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '.04em' }}>
                  Conversations
                </div>
              )}
              <div
                className={`palette-item ${i === selected ? 'selected' : ''}`}
                onClick={() => onNavigate(item.path)}
                onMouseEnter={() => setSelected(i)}
              >
                <span className="palette-item-icon">{item.icon}</span>
                <div className="palette-item-text">
                  <div className="palette-item-title">{item.label}</div>
                  {(item as any).snippet && (
                    <div className="palette-item-sub">{(item as any).snippet}</div>
                  )}
                </div>
                <span style={{ color: 'var(--text-mute)', fontSize: 11 }}>↵</span>
              </div>
            </div>
          ))}
        </div>

        {/* Footer */}
        <div style={{
          display: 'flex', gap: 16, padding: '10px 16px',
          borderTop: '1px solid var(--border-sub)',
          fontSize: 11, color: 'var(--text-mute)',
        }}>
          <span>↑↓ Navigate</span>
          <span>↵ Open</span>
          <span>Esc Close</span>
        </div>
      </div>
    </div>
  );
}
