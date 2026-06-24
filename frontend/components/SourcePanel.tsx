'use client';

import { X, FileText, Hash, TrendingUp } from 'lucide-react';

interface Source {
  filename: string;
  chunk_number: number;
  chunk_text: string;
  reranker_score: number;
  similarity: number;
}

interface Props {
  sources: Source[];
  activeIndex: number | null;
  open: boolean;
  onClose: () => void;
}

function scoreLabel(score: number): { label: string; cls: string } {
  if (score > 0)     return { label: `+${score.toFixed(2)}`,  cls: 'high' };
  if (score > -3)    return { label: score.toFixed(2),         cls: 'medium' };
  return              { label: score.toFixed(2),               cls: 'low' };
}

export default function SourcePanel({ sources, activeIndex, open, onClose }: Props) {
  // Scroll active source into view
  const scrollToActive = (node: HTMLDivElement | null) => {
    if (node && activeIndex !== null) {
      const el = node.querySelector(`[data-source-idx="${activeIndex}"]`);
      if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  };

  return (
    <div className={`source-panel ${open ? 'open' : ''}`}>
      {/* Header */}
      <div className="source-panel-header">
        <div className="source-panel-title">
          Sources
          {sources.length > 0 && (
            <span style={{ marginLeft: 8, fontSize: 11, color: 'var(--text-mute)', fontWeight: 400 }}>
              {sources.length} chunk{sources.length !== 1 ? 's' : ''} retrieved
            </span>
          )}
        </div>
        <button className="source-panel-close" onClick={onClose} title="Close source panel">
          <X size={15} />
        </button>
      </div>

      {/* Sources */}
      <div className="source-panel-body" ref={scrollToActive}>
        {sources.length === 0 ? (
          <div className="empty-state" style={{ padding: '40px 20px' }}>
            <div className="empty-icon"><FileText size={32} /></div>
            <div className="empty-sub">Ask a question to see the source chunks used to generate the answer.</div>
          </div>
        ) : (
          sources.map((src, i) => {
            const score = scoreLabel(src.reranker_score);
            const isActive = i === activeIndex;
            return (
              <div
                key={i}
                data-source-idx={i}
                className="source-card"
                style={isActive ? {
                  borderLeftColor: 'var(--primary-hi)',
                  background: 'var(--surface-hi2)',
                  boxShadow: '0 0 0 1px var(--primary-dim)',
                } : {}}
              >
                {/* Header row */}
                <div className="source-card-header">
                  <div style={{ minWidth: 0 }}>
                    <div className="source-card-file" title={src.filename}>
                      <FileText size={11} style={{ display: 'inline', marginRight: 4, opacity: 0.6 }} />
                      {src.filename}
                    </div>
                    <div className="source-card-meta" style={{ marginTop: 3 }}>
                      <Hash size={9} style={{ display: 'inline', marginRight: 2 }} />
                      chunk {src.chunk_number}
                    </div>
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 4 }}>
                    <span className={`source-card-score ${score.cls}`}>{score.label}</span>
                    <span style={{ fontSize: 10, color: 'var(--text-mute)' }}>
                      sim {(src.similarity * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>

                {/* Chunk text */}
                <div className="source-card-text">{src.chunk_text}</div>

                {/* Citation badge */}
                {isActive && (
                  <div style={{
                    marginTop: 8, display: 'inline-block',
                    background: 'var(--primary-dim)', color: 'var(--primary-hi)',
                    fontSize: 10, padding: '2px 8px', borderRadius: 10,
                    border: '1px solid var(--primary)',
                  }}>
                    [{i + 1}] cited in answer
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
