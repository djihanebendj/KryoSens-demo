"use client";

// Technician finding entry and the incident record list. A finding is the technician's own statement, entered
// verbatim; the LLM never creates or confirms one. "Prepare draft" writes only a PENDING draft. The exact backend
// draft is rendered before the single-use token can be used. "Confirm finding" is the only action that writes a
// record; "Cancel draft" writes nothing. The frontend computes nothing: every field is copied from the API.
import { useRef, useState } from "react";

import { ApiFailure } from "@/lib/api/client";
import type { FindingDraftCreated, FindingOutcome, FindingRecord, FindingDraft, IncidentView } from "@/lib/api/types";
import { useCase } from "@/lib/case-context";
import { formatTimestamp } from "@/lib/format";

import { StatusBadge } from "./badges";
import { ErrorState } from "./states";

function failure(e: unknown): ApiFailure {
  return e instanceof ApiFailure
    ? e
    : new ApiFailure(0, { code: "UNEXPECTED_RESPONSE", message: "the request failed", retryable: false });
}

function OutcomeBadge({ outcome }: { outcome: FindingRecord["outcome"] }) {
  const t = outcome === "RESOLVED" ? "good" : outcome === "NOT_RESOLVED" ? "bad" : "warn";
  return <StatusBadge value={outcome} t={t} />;
}

function RecordCard({ r, testId }: { r: FindingRecord; testId: string }) {
  return (
    <article data-testid={testId} className="space-y-2 rounded-lg border border-slate-200 bg-white p-4 text-sm">
      <header className="flex flex-wrap items-center gap-2">
        <code className="text-xs">{r.record_id}</code>
        <OutcomeBadge outcome={r.outcome} />
        <span className="text-xs text-slate-500">
          created {formatTimestamp(r.created_at)} · confirmed {formatTimestamp(r.confirmed_at)}
        </span>
      </header>
      <pre className="whitespace-pre-wrap break-words font-sans text-slate-800">{r.statement}</pre>
      <p className="text-xs text-slate-500">Event window: {r.event_window ?? "—"}</p>
      <p className="text-xs text-slate-500 break-all">Content hash: {r.content_hash}</p>
      <p className="text-xs text-slate-600">
        Confirmed by you in this web application. The Phase 8 store preserves its audit labels verbatim: source “{r.source}”,
        confirmed via “{r.confirmed_via}”.
      </p>
    </article>
  );
}

function DraftCard({ d, testId }: { d: FindingDraft; testId: string }) {
  return (
    <article data-testid={testId} className="space-y-2 rounded-lg border border-dashed border-amber-300 bg-amber-50 p-4 text-sm">
      <header className="flex flex-wrap items-center gap-2">
        <code className="text-xs">{d.draft_id}</code>
        <StatusBadge value={d.status} t={d.status === "CANCELLED" ? "neutral" : "warn"} />
        <span className="text-xs text-amber-800">
          {d.status === "CANCELLED" ? "cancelled — nothing was recorded" : "draft — not history"}
        </span>
        <span className="text-xs text-slate-500">created {formatTimestamp(d.created_at)}</span>
      </header>
      <pre className="whitespace-pre-wrap break-words font-sans text-slate-800">{d.statement}</pre>
      <p className="text-xs text-slate-500">Event window: {d.event_window ?? "—"}</p>
    </article>
  );
}

export function FindingsPanel({
  caseId,
  incident,
  onChanged,
}: {
  caseId: string;
  incident: IncidentView;
  onChanged: () => void;
}) {
  const { api } = useCase();
  const [statement, setStatement] = useState("");
  const [eventWindow, setEventWindow] = useState("");
  const [draft, setDraft] = useState<FindingDraftCreated | null>(null);
  const [outcome, setOutcome] = useState<FindingOutcome>("UNKNOWN");
  const [busy, setBusy] = useState<"draft" | "confirm" | "cancel" | null>(null);
  const [error, setError] = useState<ApiFailure | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const inFlight = useRef(false);

  async function run(kind: "draft" | "confirm" | "cancel", action: () => Promise<void>) {
    if (inFlight.current) return;                 // the token is single-use; never send two clicks at once
    inFlight.current = true;
    setBusy(kind);
    setError(null);
    try {
      await action();
    } catch (e) {
      const f = failure(e);
      if (["FINDING_CONFIRMATION_EXPIRED", "FINDING_CONFIRMATION_INVALID", "FINDING_CONFIRMATION_REPLAY",
           "FINDING_ALREADY_CONFIRMED", "FINDING_CANCELLED"].includes(f.code)) {
        setDraft(null);                           // the token can no longer be used; ask for a fresh draft
        onChanged();
      }
      setError(f);
    } finally {
      inFlight.current = false;
      setBusy(null);
    }
  }

  function prepare(e: React.FormEvent) {
    e.preventDefault();
    if (!statement.trim()) return;
    void run("draft", async () => {
      const d = await api.createFindingDraft(caseId, {
        statement,
        event_window: eventWindow.trim() ? eventWindow : null,
      });
      setDraft(d);
      setOutcome("UNKNOWN");
      setNotice("Draft prepared. Nothing is recorded until you confirm it.");
      onChanged();
    });
  }

  function confirm() {
    if (!draft) return;
    void run("confirm", async () => {
      const r = await api.confirmFinding(caseId, draft.draft_id, { token: draft.confirmation_token, outcome });
      setDraft(null);
      setStatement("");
      setEventWindow("");
      setNotice(`Finding ${r.record_id} confirmed. It is now a confirmed record, not a draft.`);
      onChanged();
    });
  }

  function cancel() {
    if (!draft) return;
    void run("cancel", async () => {
      await api.cancelFinding(caseId, draft.draft_id);
      setDraft(null);
      setNotice("Draft cancelled. Nothing was written and no finding was added.");
      onChanged();
    });
  }

  const pending = incident.drafts.filter((d) => d.status === "PENDING");
  const cancelled = incident.drafts.filter((d) => d.status === "CANCELLED");
  const unknown = incident.confirmed.filter((r) => r.outcome === "UNKNOWN");
  const verified = incident.confirmed.filter((r) => r.outcome === "RESOLVED" || r.outcome === "NOT_RESOLVED");

  return (
    <div className="space-y-6">
      <section className="rounded-lg border border-slate-200 bg-white p-5" data-testid="finding-form">
        <h2 className="text-lg font-semibold">Record a technician finding</h2>
        <p className="mt-1 text-sm text-slate-600">
          Your own observation, in your own words. It is not generated by a language model and it is not a confirmed
          fault until you explicitly confirm the exact draft. ColdSense does not infer causation from timing.
        </p>
        <form onSubmit={prepare} className="mt-4 space-y-3">
          <div>
            <label htmlFor="finding-statement" className="text-sm font-medium">
              Statement (shown and stored verbatim)
            </label>
            <textarea
              id="finding-statement"
              value={statement}
              onChange={(e) => setStatement(e.target.value)}
              maxLength={4000}
              rows={4}
              className="mt-1 w-full rounded-md border border-slate-300 p-2 text-sm"
              placeholder="What did you observe? e.g. the door seal was worn and the compressor stayed off."
            />
            <p className="mt-1 text-xs text-slate-500">{statement.length}/4000 characters. Blank statements are refused.</p>
          </div>
          <div>
            <label htmlFor="finding-window" className="text-sm font-medium">
              Event window (optional, free text)
            </label>
            <input
              id="finding-window"
              value={eventWindow}
              onChange={(e) => setEventWindow(e.target.value)}
              maxLength={200}
              className="mt-1 w-full rounded-md border border-slate-300 p-2 text-sm"
              placeholder="e.g. overnight power cut"
            />
            <p className="mt-1 text-xs text-slate-500">
              A window containing a number may make the Phase 8 report builder refuse the report rather than
              accept an untraceable value.
            </p>
          </div>
          <button
            type="submit"
            disabled={busy !== null || !statement.trim()}
            className="rounded-md bg-sky-700 px-4 py-2 text-sm font-medium text-white hover:bg-sky-800 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {busy === "draft" ? "Preparing draft…" : "Prepare draft"}
          </button>
          <p className="text-xs text-slate-500">
            Preparing a draft writes no finding. Only “Confirm finding” on the exact draft below can write one.
          </p>
        </form>
      </section>

      {error && <ErrorState error={error} />}
      {notice && (
        <p role="status" data-testid="finding-notice" className="rounded-lg border border-sky-300 bg-sky-50 p-3 text-sm text-sky-900">
          {notice}
        </p>
      )}

      {draft && (
        <section data-testid="finding-draft" className="space-y-3 rounded-lg border border-amber-400 bg-amber-50 p-5">
          <header className="flex flex-wrap items-center gap-2">
            <h2 className="text-lg font-semibold">Review this draft exactly as the backend stored it</h2>
            <StatusBadge value="DRAFT · NOT HISTORY" t="warn" />
          </header>
          <dl className="grid grid-cols-1 gap-x-6 gap-y-1 text-xs sm:grid-cols-[max-content_1fr]">
            <dt className="text-slate-500">Draft id</dt>
            <dd><code>{draft.draft_id}</code></dd>
            <dt className="text-slate-500">Case (asset id)</dt>
            <dd className="break-all"><code>{draft.asset_id}</code></dd>
            <dt className="text-slate-500">Created</dt>
            <dd>{formatTimestamp(draft.created_at)}</dd>
            <dt className="text-slate-500">Content hash</dt>
            <dd className="break-all"><code>{draft.content_hash}</code></dd>
            <dt className="text-slate-500">Token expires</dt>
            <dd>{formatTimestamp(draft.token_expires_at)}</dd>
          </dl>
          <div>
            <p className="text-xs font-semibold text-slate-600">Statement</p>
            <pre data-testid="finding-draft-statement" className="mt-1 whitespace-pre-wrap break-words font-sans text-sm text-slate-900">
              {draft.statement}
            </pre>
          </div>
          <p className="text-sm text-slate-700">Event window: {draft.event_window ?? "—"}</p>
          <div>
            <label htmlFor="finding-outcome" className="text-sm font-medium">
              Outcome (optional; defaults to UNKNOWN)
            </label>
            <select
              id="finding-outcome"
              value={outcome}
              onChange={(e) => setOutcome(e.target.value as FindingOutcome)}
              className="mt-1 block rounded-md border border-slate-300 p-2 text-sm"
            >
              <option value="UNKNOWN">UNKNOWN — not known yet</option>
              <option value="RESOLVED">RESOLVED — verified by you</option>
              <option value="NOT_RESOLVED">NOT_RESOLVED — verified as not resolved</option>
            </select>
            <p className="mt-1 text-xs text-slate-500">
              An outcome is recorded only from this explicit choice. It is never inferred from timing or from a model.
            </p>
          </div>
          <div className="flex flex-wrap gap-3">
            <button
              type="button"
              onClick={confirm}
              disabled={busy !== null}
              className="rounded-md bg-emerald-700 px-4 py-2 text-sm font-medium text-white hover:bg-emerald-800 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {busy === "confirm" ? "Confirming…" : "Confirm finding"}
            </button>
            <button
              type="button"
              onClick={cancel}
              disabled={busy !== null}
              className="rounded-md border border-slate-400 px-4 py-2 text-sm text-slate-800 hover:bg-white disabled:cursor-not-allowed disabled:opacity-50"
            >
              {busy === "cancel" ? "Cancelling…" : "Cancel draft"}
            </button>
          </div>
          <p className="text-xs text-amber-900">
            The confirmation token is single-use and expires. Confirming the same draft twice is refused and writes
            nothing the second time. Cancelling writes nothing at all.
          </p>
        </section>
      )}

      <section className="space-y-3" data-testid="incident-counts">
        <h2 className="text-lg font-semibold">Incident record</h2>
        <ul className="grid grid-cols-2 gap-2 text-sm sm:grid-cols-4">
          <li className="rounded border border-slate-200 bg-white p-3" data-testid="count-confirmed">
            <span className="block text-xs text-slate-500">Confirmed findings</span>
            <span className="text-xl font-semibold">{incident.counts.confirmed}</span>
          </li>
          <li className="rounded border border-slate-200 bg-white p-3" data-testid="count-drafts">
            <span className="block text-xs text-slate-500">Drafts (not history)</span>
            <span className="text-xl font-semibold">{incident.counts.drafts_pending + incident.counts.drafts_cancelled}</span>
          </li>
          <li className="rounded border border-slate-200 bg-white p-3" data-testid="count-unknown">
            <span className="block text-xs text-slate-500">Unknown outcome</span>
            <span className="text-xl font-semibold">{incident.counts.outcomes_unknown}</span>
          </li>
          <li className="rounded border border-slate-200 bg-white p-3" data-testid="count-verified">
            <span className="block text-xs text-slate-500">Verified outcome</span>
            <span className="text-xl font-semibold">{incident.counts.outcomes_verified}</span>
          </li>
        </ul>
      </section>

      <section className="space-y-3" data-testid="incident-confirmed">
        <h3 className="font-semibold">Confirmed findings ({incident.counts.confirmed})</h3>
        {incident.counts.confirmed === 0 && (
          <p className="text-sm text-slate-500">No finding has been confirmed yet. A draft is not history.</p>
        )}
        <div className="space-y-3" data-testid="outcome-unknown">
          <p className="text-xs font-semibold text-slate-600">Outcome unknown ({incident.counts.outcomes_unknown})</p>
          {unknown.length === 0 && <p className="text-xs text-slate-400">none</p>}
          {unknown.map((r) => <RecordCard key={r.record_id} r={r} testId={`confirmed-${r.record_id}`} />)}
        </div>
        <div className="space-y-3" data-testid="outcome-verified">
          <p className="text-xs font-semibold text-slate-600">Verified outcome ({incident.counts.outcomes_verified})</p>
          {verified.length === 0 && <p className="text-xs text-slate-400">none</p>}
          {verified.map((r) => <RecordCard key={r.record_id} r={r} testId={`confirmed-${r.record_id}`} />)}
        </div>
      </section>

      <section className="space-y-3" data-testid="incident-drafts">
        <h3 className="font-semibold">Drafts — not history ({pending.length + cancelled.length})</h3>
        <p className="text-xs text-slate-500">
          A draft is excluded from every report until it is confirmed. A cancelled draft wrote nothing.
        </p>
        {pending.length === 0 && cancelled.length === 0 && <p className="text-sm text-slate-500">No drafts.</p>}
        {pending.map((d) => <DraftCard key={d.draft_id} d={d} testId={`draft-${d.draft_id}`} />)}
        {cancelled.map((d) => <DraftCard key={d.draft_id} d={d} testId={`draft-${d.draft_id}`} />)}
      </section>
    </div>
  );
}
