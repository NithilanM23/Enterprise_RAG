'use client';

import './globals.css';
import { useState, useEffect, useCallback, useRef } from 'react';
import Link from 'next/link';
import { usePathname, useRouter } from 'next/navigation';
import {
  MessageSquare, FileText, BarChart2, Bookmark,
  Settings, Plus, Search, Trash2, Brain, X,
  MessagesSquare, ChevronRight,
} from 'lucide-react';
import { sessions as sessionsApi } from '@/utils/api';
import CommandPalette from '@/components/CommandPalette';

import { AuthProvider, useAuth } from './context/AuthContext';

function AppLayout({ children }: { children: React.ReactNode }) {
  const pathname  = usePathname();
  const router    = useRouter();
  const [sessions, setSessions]   = useState<any[]>([]);
  const [palette,  setPalette]    = useState(false);
  const { logout, username } = useAuth();

  const loadSessions = useCallback(async () => {
    try { setSessions(await sessionsApi.list()); } catch {}
  }, []);

  useEffect(() => { 
    if (pathname !== '/login') loadSessions(); 
  }, [loadSessions, pathname]);

  // Ctrl+K global shortcut
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        setPalette(p => !p);
      }
      if (e.key === 'Escape') setPalette(false);
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, []);

  const newChat = async () => {
    try {
      const s = await sessionsApi.create();
      await loadSessions();
      router.push(`/chat/${s.id}`);
    } catch {}
  };

  const deleteSession = async (e: React.MouseEvent, id: number) => {
    e.preventDefault();
    e.stopPropagation();
    try {
      await sessionsApi.delete(id);
      setSessions(prev => prev.filter(s => s.id !== id));
      if (pathname === `/chat/${id}`) router.push('/');
    } catch {}
  };

  if (pathname === '/login') {
    return <>{children}</>;
  }

  // Group sessions by time
  const now   = Date.now();
  const today = sessions.filter(s => now - new Date(s.updated_at).getTime() < 86400000);
  const week  = sessions.filter(s => {
    const age = now - new Date(s.updated_at).getTime();
    return age >= 86400000 && age < 7 * 86400000;
  });
  const older = sessions.filter(s => now - new Date(s.updated_at).getTime() >= 7 * 86400000);

  const navItems = [
    { href: '/',           icon: <Brain size={15} />,         label: 'Dashboard' },
    { href: '/documents',  icon: <FileText size={15} />,      label: 'Documents' },
    { href: '/explorer',   icon: <BarChart2 size={15} />,     label: 'Data Explorer' },
    { href: '/saved',      icon: <Bookmark size={15} />,      label: 'Saved' },
  ];

  if (username === 'admin') {
    navItems.push({ href: '/admin',      icon: <Settings size={15} />,      label: 'Admin' });
  }

  const SessionGroup = ({ label, items }: { label: string; items: any[] }) => {
    if (!items.length) return null;
    return (
      <>
        <div className="sidebar-time-group">{label}</div>
        {items.map(s => (
          <div
            key={s.id}
            className={`session-item ${pathname === `/chat/${s.id}` ? 'active' : ''}`}
            onClick={() => router.push(`/chat/${s.id}`)}
          >
            <MessagesSquare size={13} style={{ flexShrink: 0, opacity: 0.5 }} />
            <span className="session-title">{s.title}</span>
            <button
              className="session-delete"
              onClick={e => deleteSession(e, s.id)}
              title="Delete session"
            >
              <Trash2 size={11} />
            </button>
          </div>
        ))}
      </>
    );
  };

  return (
    <div className="app-shell">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <div className="sidebar-logo">
            <div className="sidebar-logo-icon">🧠</div>
            <div>
              <div className="sidebar-logo-text">Knowledge Assistant</div>
              <div className="sidebar-logo-sub">DICV · Local · Private</div>
            </div>
          </div>

          <button
            className="sidebar-search"
            onClick={() => setPalette(true)}
          >
            <Search size={13} />
            <span>Search sessions...</span>
            <kbd>Ctrl K</kbd>
          </button>
        </div>

        {/* Navigation */}
        <nav className="sidebar-nav">
          {navItems.map(item => (
            <Link
              key={item.href}
              href={item.href}
              className={`sidebar-nav-item ${pathname === item.href ? 'active' : ''}`}
            >
              {item.icon}
              {item.label}
            </Link>
          ))}
        </nav>

        {/* Sessions */}
        <div className="sidebar-sessions">
          <button className="new-chat-btn" onClick={newChat}>
            <Plus size={14} /> New Chat
          </button>

          {sessions.length === 0 && (
            <div style={{ padding: '20px 8px', textAlign: 'center', color: 'var(--text-mute)', fontSize: 12 }}>
              No conversations yet
            </div>
          )}

          <SessionGroup label="Today"     items={today} />
          <SessionGroup label="This week" items={week} />
          <SessionGroup label="Older"     items={older} />
        </div>
        
        {/* User Footer */}
        <div style={{ padding: '16px', borderTop: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div style={{ fontSize: '13px', color: 'var(--text-mute)' }}>
            User: <strong>{username}</strong>
          </div>
          <button onClick={logout} style={{ background: 'none', border: 'none', color: 'var(--text-mute)', cursor: 'pointer', fontSize: '13px' }}>
            Logout
          </button>
        </div>
      </aside>

      {/* Main content */}
      <div className="main-content">
        {children}
      </div>

      {/* Command palette */}
      {palette && (
        <CommandPalette
          onClose={() => setPalette(false)}
          onNavigate={(path) => { setPalette(false); router.push(path); }}
        />
      )}
    </div>
  );
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <title>DICV Knowledge Assistant</title>
        <meta name="description" content="Private RAG system for DICV employees" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </head>
      <body>
        <AuthProvider>
          <AppLayout>{children}</AppLayout>
        </AuthProvider>
      </body>
    </html>
  );
}
