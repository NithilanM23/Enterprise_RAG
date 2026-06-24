'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { useParams, useSearchParams, useRouter } from 'next/navigation';
import {
  Send, Bookmark, BookmarkCheck, Copy, ChevronDown, X,
  Layers, StopCircle, MessageSquare,
} from 'lucide-react';
import { sessions as sessionsApi, streamAsk, saved as savedApi, documents as documentsApi } from '@/utils/api';
import SourcePanel from '@/components/SourcePanel';

interface MessageUI {
  id?: number;
  role: 'user' | 'assistant';
  content: string;
  sources?: any[];
  is_pinned?: boolean;
  streaming?: boolean;
  followups?: string[];
  routing?: { category: string; confidence: number };
  loadingStatus?: string;
}

export default function ChatPage() {
  const params      = useParams();
  const searchParams = useSearchParams();
  const router      = useRouter();
  const sessionId   = Number(params.sessionId);

  const [messages,    setMessages]   = useState<MessageUI[]>([]);
  const [input,       setInput]      = useState('');
  const [sending,     setSending]    = useState(false);
  const [title,       setTitle]      = useState('New Chat');
  const [sources,     setSources]    = useState<any[]>([]);
  const [panelOpen,   setPanelOpen]  = useState(false);
  const [activeSource, setActiveSource] = useState<number | null>(null);
  const [streamCtrl,  setStreamCtrl] = useState<AbortController | null>(null);
  const [toast,       setToast]      = useState<string | null>(null);
  const [allDocs,     setAllDocs]    = useState<any[]>([]);
  const [selectedDocIds, setSelectedDocIds] = useState<number[] | null>(null);
  const [scopeExpanded, setScopeExpanded] = useState(false);
  const [historyLoaded, setHistoryLoaded] = useState(false);
  const [autoSent, setAutoSent] = useState(false);

  const bottomRef  = useRef<HTMLDivElement>(null);
  const inputRef   = useRef<HTMLTextAreaElement>(null);

  // Load session history
  const loadHistory = useCallback(async () => {
    try {
      documentsApi.list().then(setAllDocs).catch(() => {});
    } catch {}
    if (!sessionId) {
      setHistoryLoaded(true);
      return;
    }
    try {
      const [sess, msgs] = await Promise.all([
        sessionsApi.list().then(list => list.find(s => s.id === sessionId)),
        sessionsApi.messages(sessionId),
      ]);
      if (sess) setTitle(sess.title);
      setMessages(prev => {
        if (msgs.length === 0 && prev.length > 0) {
          return prev;
        }
        return msgs.map((m: any) => ({
          id: m.id, role: m.role, content: m.content,
          sources: m.sources, is_pinned: m.is_pinned,
        }));
      });
    } catch {} finally {
      setHistoryLoaded(true);
    }
  }, [sessionId]);

  useEffect(() => { loadHistory(); }, [loadHistory]);



  // Scroll to bottom on new messages
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const showToast = (msg: string) => {
    setToast(msg);
    setTimeout(() => setToast(null), 2500);
  };

  // Auto-resize textarea
  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
    e.target.style.height = 'auto';
    e.target.style.height = Math.min(e.target.scrollHeight, 160) + 'px';
  };

  const sendMessage = useCallback(async (overrideQ?: string) => {
    const q = (overrideQ ?? input).trim();
    if (!q || sending) return;

    setSending(true);
    setInput('');
    if (inputRef.current) inputRef.current.style.height = 'auto';

    // Optimistic user message
    const userMsg: MessageUI = { role: 'user', content: q };
    setMessages(prev => [...prev, userMsg]);

    // Streaming assistant message placeholder
    const assistantMsg: MessageUI = { role: 'assistant', content: '', streaming: true, loadingStatus: '✨ Analyzing query...' };
    setMessages(prev => [...prev, assistantMsg]);

    let routingInfo: any = null;
    let currentSources: any[] = [];

    const ctrl = streamAsk(
      sessionId, q, selectedDocIds,
      // onEvent
      (evt) => {
        if (evt.type === 'routing') {
          routingInfo = { category: evt.category, confidence: evt.confidence };
          setMessages(prev => {
            if (prev.length === 0) return prev;
            const msgs = [...prev];
            const last = msgs[msgs.length - 1];
            if (last && last.role === 'assistant') {
              msgs[msgs.length - 1] = { ...last, loadingStatus: `🔍 Searching ${evt.category.replace(/_/g, ' ')}...` };
            }
            return msgs;
          });
        }
        if (evt.type === 'sources') {
          currentSources = evt.chunks;
          setSources(evt.chunks);
          setMessages(prev => {
            if (prev.length === 0) return prev;
            const msgs = [...prev];
            const last = msgs[msgs.length - 1];
            if (last && last.role === 'assistant') {
              msgs[msgs.length - 1] = { ...last, loadingStatus: `📚 Reading ${evt.chunks.length} documents...` };
            }
            return msgs;
          });
          // After a short delay, update to formulating
          setTimeout(() => {
            setMessages(prev => {
              if (prev.length === 0) return prev;
              const msgs = [...prev];
              const last = msgs[msgs.length - 1];
              if (last && last.role === 'assistant' && last.loadingStatus) {
                msgs[msgs.length - 1] = { ...last, loadingStatus: `💡 Formulating answer...` };
              }
              return msgs;
            });
          }, 600);
        }
        if (evt.type === 'token') {
          setMessages(prev => {
            if (prev.length === 0) return prev;
            const msgs = [...prev];
            const last = msgs[msgs.length - 1];
            if (last && last.role === 'assistant') {
              const updated = { ...last, content: last.content + evt.text };
              delete updated.loadingStatus; // Clear loading status once text starts
              msgs[msgs.length - 1] = updated;
            }
            return msgs;
          });
        }
        if (evt.type === 'done') {
          setMessages(prev => {
            if (prev.length === 0) return prev;
            const msgs = [...prev];
            const last = msgs[msgs.length - 1];
            if (last && last.role === 'assistant') {
              msgs[msgs.length - 1] = {
                ...last,
                streaming: false,
                sources: currentSources,
                routing: routingInfo,
                id: evt.message_id,
              };
            }
            return msgs;
          });
          if (title === 'New Chat') {
            setTitle(q.slice(0, 60));
          }
        }
        if (evt.type === 'followups') {
          setMessages(prev => {
            const msgs = [...prev];
            const last = msgs[msgs.length - 1];
            if (last.role === 'assistant') {
              msgs[msgs.length - 1] = { ...last, followups: evt.suggestions };
            }
            return msgs;
          });
        }
        if (evt.type === 'error') {
          setMessages(prev => {
            const msgs = [...prev];
            const last = msgs[msgs.length - 1];
            if (last.role === 'assistant') {
              msgs[msgs.length - 1] = { ...last, streaming: false, content: `❌ ${evt.message}` };
            }
            return msgs;
          });
        }
      },
      // onDone
      () => { setSending(false); setStreamCtrl(null); },
      // onError
      (err) => {
        setMessages(prev => {
          const msgs = [...prev];
          const last = msgs[msgs.length - 1];
          if (last.role === 'assistant') {
            msgs[msgs.length - 1] = { ...last, streaming: false, content: `❌ ${err}` };
          }
          return msgs;
        });
        setSending(false);
        setStreamCtrl(null);
      }
    );

    setStreamCtrl(ctrl);
  }, [input, sending, sessionId, title, selectedDocIds]);

  // Auto-send if ?q= param is present
  useEffect(() => {
    if (historyLoaded && !autoSent) {
      const q = searchParams.get('q');
      if (q && messages.length === 0) {
        setAutoSent(true);
        // Strip q from URL so refresh doesn't trigger it again
        window.history.replaceState({}, '', `/chat/${sessionId}`);
        setTimeout(() => sendMessage(q), 50);
      }
    }
  }, [historyLoaded, autoSent, searchParams, messages.length, sendMessage, sessionId]);

  const stopStream = () => {
    streamCtrl?.abort();
    setStreamCtrl(null);
    setSending(false);
    setMessages(prev => {
      const msgs = [...prev];
      const last = msgs[msgs.length - 1];
      if (last.streaming) msgs[msgs.length - 1] = { ...last, streaming: false };
      return msgs;
    });
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const handlePin = async (msg: MessageUI) => {
    if (!msg.id) return;
    try {
      if (msg.is_pinned) {
        await savedApi.unpin(msg.id);
        setMessages(prev => prev.map(m => m.id === msg.id ? { ...m, is_pinned: false } : m));
        showToast('Answer unpinned.');
      } else {
        await savedApi.pin(msg.id);
        setMessages(prev => prev.map(m => m.id === msg.id ? { ...m, is_pinned: true } : m));
        showToast('Answer pinned to Saved.');
      }
    } catch {}
  };

  const handleCopy = (text: string) => {
    navigator.clipboard.writeText(text).then(() => showToast('Copied to clipboard.'));
  };

  const handleCitationClick = (sourceIdx: number) => {
    setActiveSource(sourceIdx);
    setPanelOpen(true);
  };

  // Render message content — turn [1] [2] into clickable citation badges
  const renderContent = (content: string, msgSources?: any[]) => {
    if (!msgSources?.length) return <span>{content}</span>;
    const parts = content.split(/(\[\d+\])/g);
    return (
      <>
        {parts.map((part, i) => {
          const match = part.match(/^\[(\d+)\]$/);
          if (match) {
            const idx = parseInt(match[1]) - 1;
            return (
              <span
                key={i}
                className="citation"
                onClick={() => { setSources(msgSources); handleCitationClick(idx); }}
                title={`Source ${match[1]}: ${msgSources[idx]?.filename || ''}`}
              >
                {match[1]}
              </span>
            );
          }
          return <span key={i}>{part}</span>;
        })}
      </>
    );
  };

  return (
    <>
      {/* Chat layout */}
      <div className="chat-layout">
        <div className="chat-main">
          {/* Messages */}
          <div className="chat-messages">
            {messages.length === 0 && (
              <div className="empty-state-modern">
                <div className="empty-title">Hello, how can I help?</div>
              </div>
            )}

            {messages.map((msg, i) => (
              <div key={i} className={`msg-row ${msg.role}`}>
                <div className={`msg-avatar ${msg.role}`}>
                  {msg.role === 'user' ? 'U' : '🧠'}
                </div>

                <div className="msg-content-wrapper">
                  {/* Routing badge for assistant */}
                  {msg.role === 'assistant' && msg.routing?.category && (
                    <div className="routing-badge">
                      <div className="routing-dot" />
                      Searched: {msg.routing.category.replace(/_/g, ' ')}
                      {msg.routing.confidence >= 2 ? ' (focused)' : ' (broad)'}
                    </div>
                  )}

                  <div className={`msg-bubble ${msg.role}`}>
                    {msg.loadingStatus ? (
                      <div className="thinking-loader">
                        <div className="thinking-dots">
                          <span className="dot"></span>
                          <span className="dot"></span>
                          <span className="dot"></span>
                        </div>
                        <span className="thinking-text">{msg.loadingStatus}</span>
                      </div>
                    ) : (
                      <>
                        {renderContent(msg.content, msg.sources)}
                        {msg.streaming && <span className="stream-cursor" />}
                      </>
                    )}
                  </div>

                  {/* Message actions (assistant only) */}
                  {msg.role === 'assistant' && !msg.streaming && (
                    <div className="msg-actions">
                      {msg.sources && msg.sources.length > 0 && (
                        <button
                          className="msg-action-btn"
                          onClick={() => { setSources(msg.sources!); setPanelOpen(true); setActiveSource(null); }}
                        >
                          <Layers size={11} /> {msg.sources.length} source{msg.sources.length !== 1 ? 's' : ''}
                        </button>
                      )}
                      <button className="msg-action-btn" onClick={() => handleCopy(msg.content)}>
                        <Copy size={11} /> Copy
                      </button>
                      {msg.id && (
                        <button
                          className={`msg-action-btn ${msg.is_pinned ? 'pinned' : ''}`}
                          onClick={() => handlePin(msg)}
                          title={msg.is_pinned ? 'Unpin' : 'Pin to Saved'}
                        >
                          {msg.is_pinned ? <BookmarkCheck size={11} /> : <Bookmark size={11} />}
                          {msg.is_pinned ? 'Saved' : 'Save'}
                        </button>
                      )}
                    </div>
                  )}

                  {/* Follow-up chips */}
                  {msg.role === 'assistant' && !msg.streaming && msg.followups && msg.followups.length > 0 && (
                    <div className="followup-row">
                      {msg.followups.map((q, fi) => (
                        <button
                          key={fi}
                          className="followup-chip"
                          onClick={() => { setInput(q); setTimeout(() => sendMessage(q), 50); }}
                        >
                          {q}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            ))}

            <div ref={bottomRef} />
          </div>

          {/* Input area */}
          <div className="chat-input-area">
            <div className="chat-input-wrap">
              <textarea
                ref={inputRef}
                className="chat-input"
                rows={1}
                placeholder="Ask anything about your documents… (Enter to send, Shift+Enter for new line)"
                value={input}
                onChange={handleInputChange}
                onKeyDown={handleKeyDown}
                disabled={sending}
              />
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <div style={{ position: 'relative' }}>
                  <button 
                    className={`chat-scope-btn ${selectedDocIds !== null ? 'active' : ''}`}
                    onClick={() => setScopeExpanded(!scopeExpanded)}
                    title={selectedDocIds ? `Searching ${selectedDocIds.length} docs` : 'Searching all docs'}
                  >
                    <Layers size={18} />
                    {selectedDocIds && <div className="chat-scope-dot" />}
                  </button>
                  
                  {scopeExpanded && (
                    <div className="chat-scope-dropdown">
                      <div className="chat-scope-dropdown-header">Document Scope</div>
                      <div className="chat-scope-dropdown-list">
                        <label className="chat-scope-option">
                          <input 
                            type="checkbox" 
                            checked={selectedDocIds === null} 
                            onChange={() => setSelectedDocIds(null)} 
                          />
                          <span>All documents</span>
                        </label>
                        {allDocs.map(doc => (
                          <label key={doc.id} className="chat-scope-option">
                            <input 
                              type="checkbox" 
                              checked={selectedDocIds !== null && selectedDocIds.includes(doc.id)}
                              onChange={(e) => {
                                if (e.target.checked) {
                                  setSelectedDocIds(prev => prev === null ? [doc.id] : [...prev, doc.id]);
                                } else {
                                  setSelectedDocIds(prev => {
                                    if (prev === null) return null;
                                    const next = prev.filter(id => id !== doc.id);
                                    return next.length === 0 ? null : next;
                                  });
                                }
                              }}
                            />
                            <span className="chat-scope-filename">{doc.filename}</span>
                          </label>
                        ))}
                      </div>
                    </div>
                  )}
                </div>

                {sending ? (
                  <button className="chat-send-btn" onClick={stopStream} title="Stop generating" style={{ background: 'var(--error)' }}>
                    <StopCircle size={16} />
                  </button>
                ) : (
                  <button
                    className="chat-send-btn"
                    onClick={() => sendMessage()}
                    disabled={!input.trim()}
                    title="Send (Enter)"
                  >
                    <Send size={16} />
                  </button>
                )}
              </div>
            </div>
            <div className="chat-hint" style={{ marginTop: 8 }}>
              Answers sourced exclusively from your uploaded documents.
            </div>
          </div>
        </div>

        {/* Source panel */}
        <SourcePanel
          sources={sources}
          activeIndex={activeSource}
          open={panelOpen}
          onClose={() => setPanelOpen(false)}
        />
      </div>

      {/* Toast */}
      {toast && (
        <div className="toast-container">
          <div className="toast info">{toast}</div>
        </div>
      )}
    </>
  );
}
