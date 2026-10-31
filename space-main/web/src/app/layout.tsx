import type { Metadata } from "next";

import { JourneyBar, Nav } from "@/components/nav";
import { CaseProvider } from "@/lib/case-context";

import "./globals.css";

export const metadata: Metadata = {
  title: "KryoSens",
  description: "Evidence-first decision support for cold rooms. When the evidence does not support an answer, KryoSens refuses and says why.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-slate-50 text-slate-900 antialiased">
        <CaseProvider>
          <header className="border-b border-slate-200 bg-white">
            <div className="mx-auto flex max-w-6xl flex-wrap items-baseline justify-between gap-2 px-4 py-3">
              <p className="text-lg font-semibold tracking-tight">KryoSens</p>
              <p className="text-xs text-slate-500">
                Evidence-first. Not validated on a physical cold room. No food-safety claim.
              </p>
            </div>
          </header>
          <div className="mx-auto grid max-w-6xl grid-cols-1 gap-6 px-4 py-6 md:grid-cols-[200px_1fr]">
            <Nav />
            <main className="min-w-0 space-y-4">
              <JourneyBar />
              {children}
            </main>
          </div>
        </CaseProvider>
      </body>
    </html>
  );
}
