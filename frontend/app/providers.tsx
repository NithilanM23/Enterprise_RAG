'use client';

import { SessionProvider } from "next-auth/react";

import { AppModeProvider } from "./context/AppModeContext";

export function Providers({ children }: { children: React.ReactNode }) {
  return (
    <SessionProvider>
      <AppModeProvider>
        {children}
      </AppModeProvider>
    </SessionProvider>
  );
}
