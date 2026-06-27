'use client';

import './globals.css';
import { useState, useEffect, useCallback, useRef } from 'react';
import Link from 'next/link';
import { usePathname, useRouter } from 'next/navigation';
import {
  MessageSquare, FileText, BarChart2, Bookmark,
  Settings, Plus, Search, Trash2, Brain, X,
  MessagesSquare, ChevronRight, PanelLeftClose, PanelLeftOpen
} from 'lucide-react';
import { sessions as sessionsApi } from '@/utils/api';
import CommandPalette from '@/components/CommandPalette';

import { Providers } from './providers';
import { useSession, signOut } from 'next-auth/react';

function AppLayout({ children }: { children: React.ReactNode }) {
  const pathname  = usePathname();
  const router    = useRouter();
  const [sessions, setSessions]   = useState<any[]>([]);
  const [palette,  setPalette]    = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const { data: session, status } = useSession();
  const username = session?.user?.name || '';

  useEffect(() => {
    if (status === 'unauthenticated' && pathname !== '/login') {
      router.push('/login');
    }
  }, [status, pathname, router]);

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
        {items.map(s => {
          const isActive = pathname === `/chat/${s.id}`;
          return (
            <div
              key={s.id}
              className={`session-item ${isActive ? 'active' : ''}`}
              onClick={() => router.push(`/chat/${s.id}`)}
            >
              <MessagesSquare size={14} style={{ flexShrink: 0, opacity: 0.5 }} />
              <span className="session-title">{s.title}</span>
              <button
                className="session-delete"
                onClick={e => deleteSession(e, s.id)}
                title="Delete session"
              >
                <Trash2 size={13} />
              </button>
              {isActive && <div className="session-indicator" />}
            </div>
          );
        })}
      </>
    );
  };

  return (
    <div className="app-shell">
      {/* Sidebar */}
      <aside className={`sidebar ${sidebarOpen ? '' : 'collapsed'}`}>
        <div className="sidebar-header">
          <div className="sidebar-logo-text">CHAT A.I+</div>
        </div>

        <div className="sidebar-action-row">
          <button className="new-chat-btn" onClick={newChat}>
            <Plus size={14} /> New chat
          </button>
          <button className="sidebar-search-btn" onClick={() => setPalette(true)} title="Search sessions (Ctrl+K)">
            <Search size={14} />
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

        <div className="sidebar-section-header">
          <div className="sidebar-section-label">Your conversations</div>
        </div>

        {/* Sessions */}
        <div className="sidebar-sessions">
          {sessions.length === 0 && (
            <div style={{ padding: '20px 8px', textAlign: 'center', color: 'var(--text-mute)', fontSize: 12 }}>
              No conversations yet
            </div>
          )}

          <SessionGroup label="Today"     items={today} />
          <SessionGroup label="This week" items={week} />
          <SessionGroup label="Older"     items={older} />
        </div>
        
        {/* Footer */}
        <div className="sidebar-footer">
          {username === 'admin' && (
            <button className="footer-pill" onClick={() => router.push('/admin')}>
              <Settings size={15} /> Settings
            </button>
          )}
          <button className="footer-pill" onClick={() => signOut()} title="Logout">
            <div style={{ width: 22, height: 22, borderRadius: '50%', background: 'var(--text-sec)', color: '#fff', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 11, fontWeight: 600 }}>
              {username ? username[0].toUpperCase() : 'U'}
            </div>
            {username}
          </button>
        </div>
      </aside>

      {/* Main content */}
      <div className="main-content">
        <button 
          className="sidebar-toggle-btn"
          onClick={() => setSidebarOpen(!sidebarOpen)}
          title={sidebarOpen ? "Close sidebar" : "Open sidebar"}
        >
          {sidebarOpen ? <PanelLeftClose size={18} /> : <PanelLeftOpen size={18} />}
        </button>
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
        <Providers>
          <AppLayout>{children}</AppLayout>
        </Providers>
      </body>
    </html>
  );
}
