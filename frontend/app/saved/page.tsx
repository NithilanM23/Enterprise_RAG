'use client';

import { useState, useEffect, useCallback } from 'react';
import { Bookmark, BookmarkX, ExternalLink, FileText, MessageSquare } from 'lucide-react';
import { saved as savedApi } from '@/utils/api';
import Link from 'next/link';

export default function SavedPage() {
  const [pins,  setPins]  = useState<any[]>([]);
  const [toast, setToast] = useState<string | null>(null);

  const load = useCallback(async () => {
    try { setPins(await savedApi.list()); } catch {}
  }, []);

  useEffect(() => { load(); }, [load]);

  const showToast = (msg: string) => {
    setToast(msg);
    setTimeout(() => setToast(null), 2500);
  };

  const handleUnpin = async (messageId: number) => {
    try {
      await savedApi.unpin(messageId);
      setPins(prev => prev.filter(p => p.message_id !== messageId));
      showToast('Answer unpinned.');
    } catch {}
  };

  return (
    <>
      {/* Topbar */}
      <div className="topbar">
        <Bookmark size={15} style={{ color: 'var(--primary)' }} />
        <span className="topbar-title">Saved Answers</span>
        <span className="topbar-subtitle">{pins.length} saved</span>
      </div>

      <div className="page-content">
        {pins.length === 0 ? (
          <div className="empty-state" style={{ marginTop: '15vh' }}>
            <div className="empty-icon"><Bookmark size={40} /></div>
            <div className="empty-title">No saved answers yet</div>
            <div className="empty-sub">
              Click the bookmark icon on any assistant answer in a chat
              to save it here for quick reference.
            </div>
          </div>
        ) : (
          pins.map(pin => (
            <div key={pin.pin_id} className="card" style={{ marginBottom: 0 }}>
              {/* Session context */}
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
                <MessageSquare size={12} style={{ color: 'var(--text-mute)' }} />
                <Link
                  href={`/chat/${pin.session_id}`}
                  style={{ fontSize: 12, color: 'var(--primary-hi)', display: 'flex', alignItems: 'center', gap: 4 }}
                >
                  {pin.session_title}
                  <ExternalLink size={10} />
                </Link>
                <span style={{ marginLeft: 'auto', fontSize: 11, color: 'var(--text-mute)' }}>
                  {new Date(pin.pinned_at).toLocaleDateString()}
                </span>
              </div>

              {/* Answer text */}
              <div style={{ fontSize: 13, color: 'var(--text)', lineHeight: 1.65, marginBottom: 12, whiteSpace: 'pre-wrap' }}>
                {pin.content}
              </div>

              {/* Sources */}
              {pin.sources && pin.sources.length > 0 && (
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 12 }}>
                  {pin.sources.map((src: any, i: number) => (
                    <span key={i} style={{
                      fontSize: 11, padding: '2px 8px', borderRadius: 10,
                      background: 'var(--surface-hi2)', color: 'var(--text-mute)',
                      border: '1px solid var(--border-sub)', display: 'flex', alignItems: 'center', gap: 4,
                    }}>
                      <FileText size={9} /> {src.filename}
                    </span>
                  ))}
                </div>
              )}

              {/* Note */}
              {pin.note && (
                <div style={{
                  fontSize: 12, color: 'var(--text-sec)', fontStyle: 'italic',
                  padding: '6px 10px', background: 'var(--surface-hi2)', borderRadius: 'var(--r-sm)',
                  marginBottom: 12,
                }}>
                  📝 {pin.note}
                </div>
              )}

              {/* Actions */}
              <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
                <Link href={`/chat/${pin.session_id}`} className="btn btn-ghost btn-sm">
                  <ExternalLink size={11} /> Open chat
                </Link>
                <button className="btn btn-danger btn-sm" onClick={() => handleUnpin(pin.message_id)}>
                  <BookmarkX size={11} /> Remove
                </button>
              </div>
            </div>
          ))
        )}
      </div>

      {toast && (
        <div className="toast-container">
          <div className="toast">{toast}</div>
        </div>
      )}
    </>
  );
}
