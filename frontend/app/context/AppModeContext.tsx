'use client';

import React, { createContext, useContext, useState, useEffect } from 'react';

type AppMode = 'rag' | 'llm';

interface AppModeContextType {
  appMode: AppMode;
  setAppMode: (mode: AppMode) => void;
}

const AppModeContext = createContext<AppModeContextType | undefined>(undefined);

export function AppModeProvider({ children }: { children: React.ReactNode }) {
  const [appMode, setAppModeState] = useState<AppMode>('rag');

  useEffect(() => {
    const stored = localStorage.getItem('appMode');
    if (stored === 'rag' || stored === 'llm') {
      setAppModeState(stored);
    }
  }, []);

  const setAppMode = (mode: AppMode) => {
    setAppModeState(mode);
    localStorage.setItem('appMode', mode);
  };

  return (
    <AppModeContext.Provider value={{ appMode, setAppMode }}>
      {children}
    </AppModeContext.Provider>
  );
}

export function useAppMode() {
  const context = useContext(AppModeContext);
  if (context === undefined) {
    throw new Error('useAppMode must be used within an AppModeProvider');
  }
  return context;
}
