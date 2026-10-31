"use client";

// Reports: only the Phase 8 report builder's output is shown, as plain text. It is composed from confirmed
// records only; drafts are listed as excluded and never as history. A refused report has no text: the screen shows
// the numeric-check violations instead. The frontend computes no value and renders no HTML.
import { useEffect, useRef, useState } from "react";

import { StatusBadge } from "@/components/badges";
import { ProvenancePanel } from "@/components/provenance";
import { EmptyState, ErrorState, LoadingState } from "@/components/states";
import { ApiFailure } from "@/lib/api/client";
import type { ReportView } from "@/lib/api/types";
import { useCase } from "@/lib/case-context";
import { formatTimestamp } from "@/lib/format";

function outcomeTone(outcome: string) {
  return outcome === "RESOLVED" ? "good" : outcome === "NOT_RESOLVED" ? "bad" : "warn";
}

export function ReportsClient() {
  const { caseId } = useCase();
  // Keyed by the case: a different case starts from fresh report state without resetting it inside an effect.
  return <ReportsCase key={caseId ?? "no-case"} caseId={caseId} />;
}

function ReportsCase({ caseId }: { caseId: string | null }) {
  const { api } = useCase();
  const [report, setReport] = useState<ReportView | null>(null);
  const [loaded, setLoaded] = useState(caseId === null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<ApiFailure | null>(null);
  const inFlight = useRef(false);

  useEffect(() => {
    if (!caseId) return;
    let live = true;
    api.getLatestReport(caseId).then(
      (r) => { if (live) { setReport(r); setLoaded(true); } },
      (e: unknown) => {
        if (!live) return;
        if (e instanceof ApiFailure && e.code === "REPORT_NOT_GENERATED") {
          setReport(null);                 // a normal empty state, not an error
        } else {
          setError(e instanceof ApiFailure ? e : new ApiFailure(0, {
            code: "UNEXPECTED_RESPONSE", message: "the request failed", retryable: false,
          }));
        }
        setLoaded(true);
      },
    );
    return () => { live = false; };
  }, [api, caseId]);

  async function generate() {
    if (!caseId || inFlight.current) return;
    inFlight.current = true;
    setBusy(true);
    setError(null);
    try {
      setReport(await api.createReport(caseId));
    } catch (e) {
      setError(e instanceof ApiFailure ? e : new ApiFailure(0, {
        code: "UNEXPECTED_RESPONSE", message: "the request failed", retryable: false,
      }));
    } finally {
      inFlight.current = false;
      setBusy(false);
    }
  }

  if (!caseId)
    return (
      <EmptyState title="No data uploaded yet" action={{ href: "/upload", label: "Upload Data" }}>
        Upload a log or use a bundled sample, record findings on Incident, then generate the report.
      </EmptyState>
    );
  if (!loaded) return <LoadingState label="Loading the latest report…" />;
  if (error) return <ErrorState error={error} />;

  return (
    <div className="space-y-4">
      <header className="space-y-1">
        <h1 className="text-2xl font-semibold tracking-tight">Reports</h1>
        <p className="text-sm text-slate-600">
          Built by the application from <strong>confirmed findings only</strong>. Technician statements are shown
          verbatim and labelled USER_REPORTED; a draft is listed as excluded, never as history. Every number passes
          the deterministic numeric-claim check or the report is refused.
        </p>
      </header>

      <div className="flex flex-wrap items-center gap-3">
        <button
          type="button"
          onClick={generate}
          disabled={busy}
          className="rounded-md bg-sky-700 px-4 py-2 text-sm font-medium text-white hover:bg-sky-800 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {busy ? "Building report…" : report ? "Regenerate report" : "Generate report"}
        </button>
        {report && (
          <span className="text-xs text-slate-600" data-testid="report-meta">
            Generated {formatTimestamp(report.generated_at)} · numeric check{" "}
            <StatusBadge value={report.numeric_check} t={report.numeric_check === "PASSED" ? "good" : "bad"} /> ·{" "}
            {report.n_confirmed} confirmed · {report.n_drafts_excluded} draft(s) excluded
          </span>
        )}
      </div>

      {!report && (
        <EmptyState title="No report has been generated for this case">
          Generate one after confirming at least one finding on Incident. A report with no confirmed findings is
          possible, but it states that none are confirmed.
        </EmptyState>
      )}

      {report && report.status === "REFUSED" && (
        <section role="alert" data-testid="report-refused" className="rounded-lg border border-rose-300 bg-rose-50 p-5 text-sm">
          <h2 className="font-semibold text-rose-900">REPORT REFUSED — no report text was emitted</h2>
          <p className="mt-1 text-rose-900">
            The Phase 8 numeric-claim check could not trace every number in the composed report to a confirmed record
            or an application-generated value. The refusal is the safe result; nothing was guessed or rounded away.
          </p>
          <ul className="mt-2 list-disc space-y-1 pl-5 font-mono text-xs text-rose-900">
            {report.violations.map((v, i) => <li key={i}>{v}</li>)}
          </ul>
        </section>
      )}

      {report && report.status === "OK" && (
        <>
          <section className="rounded-lg border border-slate-200 bg-white p-5">
            <h2 className="text-lg font-semibold">Service report</h2>
            <p className="mt-1 text-xs text-slate-500">
              Plain text exactly as the Phase 8 builder produced it. No HTML is rendered.
            </p>
            <pre
              data-testid="report-text"
              className="mt-3 whitespace-pre-wrap break-words rounded-md bg-slate-50 p-4 font-mono text-xs text-slate-900"
            >
              {report.report ?? ""}
            </pre>
          </section>

          <section className="space-y-2" data-testid="report-confirmed">
            <h2 className="font-semibold">Confirmed findings included ({report.confirmed.length})</h2>
            {report.confirmed.length === 0 && <p className="text-sm text-slate-500">None confirmed.</p>}
            {report.confirmed.map((r) => (
              <article key={r.record_id} data-testid={`report-record-${r.record_id}`} className="rounded border border-emerald-200 bg-white p-3 text-sm">
                <header className="flex flex-wrap items-center gap-2">
                  <code className="text-xs">{r.record_id}</code>
                  <StatusBadge value={r.outcome} t={outcomeTone(r.outcome)} />
                  <span className="text-xs text-slate-500">confirmed {formatTimestamp(r.confirmed_at)}</span>
                  <span className="text-xs font-medium text-emerald-800">CONFIRMED — included</span>
                </header>
                <pre className="mt-1 whitespace-pre-wrap break-words font-sans">{r.statement}</pre>
              </article>
            ))}
          </section>

          <section className="space-y-2" data-testid="report-drafts">
            <h2 className="font-semibold">Drafts excluded ({report.drafts_excluded.length})</h2>
            <p className="text-xs text-slate-500">Drafts are not history and are not in the report body.</p>
            {report.drafts_excluded.length === 0 && <p className="text-sm text-slate-500">No drafts excluded.</p>}
            {report.drafts_excluded.map((d) => (
              <article key={d.draft_id} data-testid={`report-draft-${d.draft_id}`} className="rounded border border-dashed border-amber-300 bg-amber-50 p-3 text-sm">
                <header className="flex flex-wrap items-center gap-2">
                  <code className="text-xs">{d.draft_id}</code>
                  <StatusBadge value={d.status} t={d.status === "CANCELLED" ? "neutral" : "warn"} />
                  <span className="text-xs font-medium text-amber-900">DRAFT — excluded from history</span>
                </header>
                <pre className="mt-1 whitespace-pre-wrap break-words font-sans">{d.statement}</pre>
              </article>
            ))}
          </section>
        </>
      )}

      {report && (
        <section className="space-y-2" data-testid="report-provenance">
          <h2 className="font-semibold">What this report is</h2>
          <p className="text-sm text-slate-600">
            Confirmed technician statements, preserved verbatim and labelled USER_REPORTED. They are not measurements
            and not a confirmed cause. The Phase 8 audit labels are preserved exactly; in this deployment the trusted
            confirming channel is the web application, not a CLI.
          </p>
          <ProvenancePanel p={report.provenance} />
        </section>
      )}
    </div>
  );
}
