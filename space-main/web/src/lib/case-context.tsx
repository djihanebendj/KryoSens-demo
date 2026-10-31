"use client";

// One case shared by every page. The case id is kept in sessionStorage (per tab); ownership is enforced by the
// backend through the HttpOnly session cookie, so the id alone grants nothing.
import { createContext, useCallback, useContext, useEffect, useMemo, useState, useSyncExternalStore } from "react";

import { api as defaultApi, ApiFailure, CASE_GONE, type Api } from "./api/client";
import type { CaseStatus, PredictionResponse } from "./api/types";

const KEY = "coldsense.caseId";

// ---- the stored case id, as an external store (sessionStorage + in-memory fallback when storage is blocked)
const listeners = new Set<() => void>();
let memoryId: string | null = null;

function readStored(): string | null {
  try {
    return window.sessionStorage.getItem(KEY);
  } catch {
    return memoryId;
  }
}

function writeStored(id: string | null) {
  memoryId = id;
  try {
    if (id) window.sessionStorage.setItem(KEY, id);
    else window.sessionStorage.removeItem(KEY);
  } catch {
    // storage unavailable (private mode): the in-memory id lasts for this page view only
  }
  listeners.forEach((l) => l());
}

function subscribe(l: () => void) {
  listeners.add(l);
  return () => void listeners.delete(l);
}

// ---- context
type Phase = "empty" | "loading" | "ready" | "error";

export type CaseContextValue = {
  api: Api;
  caseId: string | null;
  status: CaseStatus | null;
  phase: Phase;
  error: ApiFailure | null;
  prediction: PredictionResponse | null;
  setStatus: (s: CaseStatus) => void;
  setPrediction: (p: PredictionResponse | null) => void;
  refresh: () => Promise<void>;
  clear: () => void;
};

type Loaded = { id: string; status: CaseStatus | null; error: ApiFailure | null };

const Ctx = createContext<CaseContextValue | null>(null);

function asFailure(e: unknown): ApiFailure {
  return e instanceof ApiFailure ? e : new ApiFailure(0, { code: "UNEXPECTED_RESPONSE", message: String(e), retryable: false });
}

export function CaseProvider({ children, api = defaultApi }: { children: React.ReactNode; api?: Api }) {
  // undefined while rendering on the server / before hydration: the stored id is not known yet
  const stored = useSyncExternalStore(subscribe, readStored, () => undefined);
  const [loaded, setLoaded] = useState<Loaded | null>(null);
  const [goneError, setGoneError] = useState<ApiFailure | null>(null);
  const [prediction, setPredictionState] = useState<{ id: string; p: PredictionResponse } | null>(null);

  const fetchStatus = useCallback(
    (id: string) =>
      api.getCase(id).then(
        (status) => setLoaded({ id, status, error: null }),
        (e) => {
          const f = asFailure(e);
          if (CASE_GONE.has(f.code)) {
            setGoneError(f);
            writeStored(null);
          } else {
            setLoaded({ id, status: null, error: f });
          }
        },
      ),
    [api],
  );

  useEffect(() => {
    if (stored) void fetchStatus(stored);
  }, [stored, fetchStatus]);

  const caseId = stored ?? null;
  const current = loaded && loaded.id === caseId ? loaded : null;
  const phase: Phase =
    stored === undefined ? "loading"
    : stored === null ? (goneError ? "error" : "empty")
    : current === null ? "loading"
    : current.error ? "error"
    : "ready";

  const value = useMemo<CaseContextValue>(
    () => ({
      api,
      caseId,
      status: current?.status ?? null,
      phase,
      error: phase === "error" ? (current?.error ?? goneError) : null,
      prediction: prediction && prediction.id === caseId ? prediction.p : null,
      setPrediction(p) {
        setPredictionState(p && caseId ? { id: caseId, p } : null);
      },
      setStatus(s) {
        setGoneError(null);
        setLoaded({ id: s.case_id, status: s, error: null });
        setPredictionState(null);
        writeStored(s.case_id);
      },
      async refresh() {
        if (caseId) await fetchStatus(caseId);
      },
      clear() {
        setGoneError(null);
        setLoaded(null);
        setPredictionState(null);
        writeStored(null);
      },
    }),
    [api, caseId, current, phase, goneError, prediction, fetchStatus],
  );
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useCase(): CaseContextValue {
  const v = useContext(Ctx);
  if (!v) throw new Error("useCase must be used inside <CaseProvider>");
  return v;
}

/** Test hook: forget the in-memory fallback id. */
export function resetCaseStoreForTests() {
  memoryId = null;
}
