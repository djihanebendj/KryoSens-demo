"use client";

import { useCallback, useEffect, useState } from "react";

import { ApiFailure, CASE_GONE } from "./api/client";
import { useCase } from "./case-context";
import { hasAnalysis } from "./journey";

export type Resource<T> =
  | { state: "no-case" }
  | { state: "loading" }
  | { state: "not-run" }
  | { state: "error"; error: ApiFailure; retry: () => void }
  | { state: "ready"; data: T };

type Result<T> = { key: string; caseId: string | null; data?: T; error?: ApiFailure };

/** Loads one view of the current case; re-loads when the case changes, on retry, or when ``reloadKey`` changes. */
/** ``needsAnalysis``: the view exists only after POST .../analysis; while the case status says it has not run,
 * nothing is requested (no request that is known to fail) and the state is "not-run". */
export function useCaseResource<T>(load: (caseId: string) => Promise<T>, reloadKey: string | number = 0, needsAnalysis = false): Resource<T> {
  const { caseId, phase, error: caseError, refresh, status } = useCase();
  const waiting = needsAnalysis && !hasAnalysis(status);
  const [result, setResult] = useState<Result<T> | null>(null);
  const [attempt, setAttempt] = useState(0);
  const retry = useCallback(() => setAttempt((n) => n + 1), []);
  const key = `${caseId}#${attempt}#${reloadKey}`;

  useEffect(() => {
    if (!caseId || phase !== "ready" || waiting) return;
    let alive = true;
    load(caseId).then(
      (data) => alive && setResult({ key, caseId, data }),
      (e) => {
        if (!alive) return;
        const f = e instanceof ApiFailure ? e : new ApiFailure(0, { code: "UNEXPECTED_RESPONSE", message: String(e), retryable: false });
        setResult({ key, caseId, error: f });
        if (CASE_GONE.has(f.code)) void refresh(); // lets the shared context drop a case that no longer exists
      },
    );
    return () => {
      alive = false;
    };
    // `load` is an inline arrow in callers; the case, its phase and the retry attempt decide a reload
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, phase, waiting]);

  if (phase === "error" && caseError) return { state: "error", error: caseError, retry: () => void refresh() };
  if (phase === "empty") return { state: "no-case" };
  if (phase === "ready" && waiting) return { state: "not-run" };
  if (phase === "loading" || !result) return { state: "loading" };
  if (result.key !== key) {
    // A refresh of the same case keeps the last result visible (the page does not unmount its state, e.g. a
    // pending finding draft). A different case never shows the previous case's data.
    if (result.caseId === caseId && result.data !== undefined) return { state: "ready", data: result.data };
    return { state: "loading" };
  }
  if (result.error) return { state: "error", error: result.error, retry };
  return { state: "ready", data: result.data as T };
}
