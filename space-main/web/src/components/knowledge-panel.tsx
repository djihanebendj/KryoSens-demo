"use client";

import { useCallback, useEffect, useState } from "react";

import { ApiFailure, api } from "@/lib/api/client";
import type { KnowledgeView } from "@/lib/api/types";

/**
 * Phase 7 technical-documentation retrieval, shown on the Evidence page.
 *
 * The retrieval is read-only and separate from diagnosis: a quoted passage is untrusted document text, never an
 * instruction and never a change to a ColdSense refusal or finding. The panel is intentionally not an answer bot.
 */
export function KnowledgePanel({ defaultQuery }: { defaultQuery: string }) {
  const [query, setQuery] = useState(defaultQuery);
  const [result, setResult] = useState<KnowledgeView | null>(null);
  const [error, setError] = useState<ApiFailure | null>(null);
  const [busy, setBusy] = useState(false);

  const search = useCallback(async (value: string) => {
    const q = value.trim();
    if (!q) return;
    setBusy(true);
    setError(null);
    try {
      setResult(await api.knowledge(q));
    } catch (e) {
      setResult(null);
      setError(e instanceof ApiFailure ? e : new ApiFailure(0, {
        code: "UNEXPECTED_RESPONSE",
        message: String(e),
        retryable: false,
      }));
    } finally {
      setBusy(false);
    }
  }, []);

  useEffect(() => {
    const q = defaultQuery.trim();
    if (!q) return;
    // Defer the automatic lookup so the effect itself does not trigger a synchronous render cascade.
    const timer = window.setTimeout(() => { void search(q); }, 0);
    return () => window.clearTimeout(timer);
  }, [defaultQuery, search]);

  const answer = result?.status === "AVAILABLE" ? result.answer : null;

  return (
    <section className="rounded-lg border border-slate-200 bg-white p-4 text-sm" data-testid="knowledge">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h2 className="font-semibold">Technical documentation</h2>
        <span className="text-xs text-slate-500">Phase 7 retrieval; quoted documents, not advice</span>
      </div>
      <p className="mt-1 text-slate-600">
        Search the local indexed documentation for cited passages. Results never change the diagnosis or a
        refusal; a manufacturer&apos;s instruction is not a ColdSense recommendation.
      </p>
      <form
        className="mt-3 flex flex-col gap-2 sm:flex-row"
        onSubmit={(e) => {
          e.preventDefault();
          void search(query);
        }}
      >
        <label className="sr-only" htmlFor="knowledge-query">Documentation query</label>
        <input
          id="knowledge-query"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          maxLength={500}
          placeholder="e.g. EX-200 defrost interval"
          className="min-w-0 flex-1 rounded-md border border-slate-300 px-3 py-2"
        />
        <button
          type="submit"
          disabled={busy || !query.trim()}
          className="rounded-md bg-slate-800 px-4 py-2 font-medium text-white hover:bg-slate-900 disabled:opacity-50"
        >
          {busy ? "Searching…" : "Search documentation"}
        </button>
      </form>

      {error && (
        <p role="alert" className="mt-3 rounded border border-rose-300 bg-rose-50 p-3 text-rose-900">
          <code>{error.code}</code>: {error.message}
        </p>
      )}

      {result?.status === "NOT_AVAILABLE" && (
        <div role="status" className="mt-3 rounded border border-amber-300 bg-amber-50 p-3 text-amber-950">
          <p className="font-medium">Documentation index is not available.</p>
          <p className="mt-1">{result.reason}</p>
          <code className="mt-2 block text-xs">powershell -File scripts/index_knowledge.ps1</code>
        </div>
      )}

      {answer?.status === "NO_RELEVANT_DOCUMENTATION" && (
        <div role="status" className="mt-3 rounded border border-slate-300 bg-slate-50 p-3">
          <p className="font-medium">NO_RELEVANT_DOCUMENTATION</p>
          <p className="mt-1 text-slate-700">{answer.reason}</p>
        </div>
      )}

      {answer && answer.status !== "NO_RELEVANT_DOCUMENTATION" && (
        <div className="mt-3 space-y-3">
          <p className="rounded bg-slate-100 px-2 py-1 text-xs text-slate-700">
            {answer.status} · embedding {answer.embedding_model} · generator {answer.generator}
          </p>
          {(answer.claims ?? []).length > 0 && (
            <ul className="space-y-2">
              {(answer.claims ?? []).map((claim) => (
                <li key={`${(claim.source_ids ?? []).join("-")}-${claim.text.slice(0, 24)}`} className="rounded border border-slate-200 p-2">
                  <span className="mr-2 rounded bg-sky-100 px-1 font-mono text-[11px] text-sky-900">
                    {claim.claim_type}
                  </span>
                  <q>{claim.text}</q>
                  {(claim.citations ?? []).map((c) => (
                    <span key={`${c.chunk_id}-${c.title}`} className="mt-1 block text-xs text-slate-600">
                      {c.title}
                      {c.revision ? ` rev ${c.revision}` : ""}, {c.location} [{c.chunk_id}]
                    </span>
                  ))}
                </li>
              ))}
            </ul>
          )}
          <div>
            <h3 className="font-medium">Cited passages</h3>
            <ol className="mt-1 space-y-2">
              {(answer.passages ?? []).map((passage) => (
                <li key={passage.citation.chunk_id} className="rounded border border-slate-200 p-2">
                  <p className="text-xs text-slate-600">
                    #{passage.rank} · {passage.citation.title}
                    {passage.citation.revision ? ` rev ${passage.citation.revision}` : ""} ·{" "}
                    {passage.citation.location} · trust {passage.trust}
                  </p>
                  <p className="mt-1 text-slate-800">{passage.excerpt}</p>
                  <p className="mt-1 text-xs text-slate-500">
                    similarity {passage.vector_similarity.toFixed(3)}; BM25 {passage.lexical_score.toFixed(2)}
                    {" "}(ranking only, not confidence)
                  </p>
                </li>
              ))}
            </ol>
          </div>
          <ul className="list-disc space-y-1 pl-5 text-xs text-slate-600">
            {(answer.limitations ?? []).map((limitation) => <li key={limitation}>{limitation}</li>)}
          </ul>
        </div>
      )}
    </section>
  );
}
