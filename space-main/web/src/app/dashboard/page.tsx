"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { StepBadge, VerdictBadge } from "@/components/badges";
import { ProvenanceLine } from "@/components/provenance";
import { EmptyState, ErrorState, LoadingState } from "@/components/states";
import { ApiFailure } from "@/lib/api/client";
import { isRefusal, type AnalysisView, type Health } from "@/lib/api/types";
import { useCase } from "@/lib/case-context";
import { formatMinutes, formatNumber, formatTimestamp, shortHash } from "@/lib/format";

export default function Dashboard() {
  const { api, caseId, status, phase, error, prediction, clear } = useCase();
  const [health, setHealth] = useState<Health | null>(null);
  const [deleteError, setDeleteError] = useState<ApiFailure | null>(null);
  const [analysisFor, setAnalysisFor] = useState<{ id: string; a: AnalysisView } | null>(null);
  const analysed = status?.steps.analysis === "READY" || status?.steps.analysis === "REFUSED";
  const analysis = analysisFor && analysisFor.id === caseId && analysed ? analysisFor.a : null;

  useEffect(() => {
    if (!caseId || !analysed) return;
    api.getAnalysis(caseId).then((a) => setAnalysisFor({ id: caseId, a }), () => undefined);
  }, [api, caseId, analysed]);

  useEffect(() => {
    api.health().then(setHealth, () => setHealth(null));
  }, [api]);

  async function remove() {
    if (!caseId) return;
    try {
      await api.deleteCase(caseId);
      clear();
    } catch (e) {
      if (e instanceof ApiFailure) setDeleteError(e);
    }
  }

  return (
    <div className="space-y-4">
      <h1 className="text-xl font-semibold">Dashboard</h1>
      {phase === "loading" && <LoadingState label="Loading the case…" />}
      {phase === "error" && error && <ErrorState error={error} />}
      {phase === "empty" && (
        <EmptyState title="No case yet" action={{ href: "/upload", label: "Upload Data" }}>
          Upload a cold-room temperature log. ColdSense checks whether the data is good enough, and forecasts only when the evidence supports it.
        </EmptyState>
      )}
      {phase === "ready" && status && (
        <>
          <section className="rounded-lg border border-slate-200 bg-white p-5" data-testid="case-card">
            <div className="flex flex-wrap items-center gap-2">
              <h2 className="break-all text-lg font-semibold">{status.filename}</h2>
            </div>
            <ProvenanceLine evidence={status.evidence_class} scope={status.validation_scope} declared={status.declared_evidence} basis={status.evidence_basis} />
            <p className="mt-1 text-xs text-slate-500">
              {status.size_bytes} bytes · sha256 {shortHash(status.sha256)} · case deleted after {formatTimestamp(status.expires_at)}
            </p>
            <table className="mt-4 w-full text-sm">
              <tbody>
                {(
                  [
                    ["Data Quality", "/quality", status.steps.quality],
                    ["Analysis (Temperature Timeline)", "/timeline", status.steps.analysis],
                    ["Prediction", "/prediction", status.steps.prediction],
                  ] as const
                ).map(([label, href, state]) => (
                  <tr key={href} className="border-t border-slate-100">
                    <td className="py-2">
                      <Link href={href} className="text-sky-800 underline-offset-2 hover:underline">
                        {label}
                      </Link>
                    </td>
                    <td className="py-2 text-right">
                      <StepBadge value={state} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
          <section className="rounded-lg border border-slate-200 bg-white p-5 text-sm">
            <div className="flex items-center gap-2">
              <h2 className="font-semibold">Data quality</h2>
              <VerdictBadge value={status.quality.verdict} />
            </div>
            <p className="mt-2 text-slate-700">
              {status.quality.rows_usable} of {status.quality.rows_read} rows usable · {formatNumber(status.quality.duration_hours, 1)} h ·{" "}
              {formatNumber(status.quality.coverage_pct, 1)} % coverage · {status.quality.blocking_issues} blocking, {status.quality.warnings} warning(s)
            </p>
          </section>
          <section className="rounded-lg border border-slate-200 bg-white p-5 text-sm" data-testid="dashboard-prediction">
            <h2 className="font-semibold">Latest prediction</h2>
            {!status.last_prediction_outcome && <p className="mt-2 text-slate-600">None requested yet.</p>}
            {prediction && isRefusal(prediction) && (
              <p className="mt-2 text-amber-900">
                Refused: <code>{prediction.refusal.code}</code> — {prediction.refusal.reason}
              </p>
            )}
            {prediction && !isRefusal(prediction) && (
              <p className="mt-2 text-slate-700">
                Forecast over {formatMinutes(prediction.horizon_min)} with <code>{prediction.model.name}</code>; crossing{" "}
                <code>{prediction.crossing.status}</code>
                {prediction.reliability && <> · reliability {prediction.reliability.status} (rule-based, not a probability)</>}
              </p>
            )}
            {status.last_prediction_outcome && !prediction && (
              <p className="mt-2 text-slate-600">
                Last outcome {status.last_prediction_outcome}. <Link href="/prediction" className="text-sky-800 underline">Open Prediction</Link>
              </p>
            )}
          </section>
          <section className="rounded-lg border border-slate-200 bg-white p-5 text-sm" data-testid="dashboard-checks">
            <h2 className="font-semibold">Outstanding recommended checks</h2>
            {!analysis && <p className="mt-2 text-slate-600">No analysis yet: run it on <Link href="/timeline" className="text-sky-800 underline">Temperature Timeline</Link>.</p>}
            {analysis && (
              <p className="mt-2 text-slate-700">
                Diagnostics <code>{analysis.diagnosis.status}</code>
                {analysis.diagnosis.refusal_reason && ` — ${analysis.diagnosis.refusal_reason}`} · {analysis.diagnosis.next_checks} check(s)
                {analysis.diagnosis.next_best_check && (
                  <>
                    ; next best check <code>{analysis.diagnosis.next_best_check}</code> (<Link href="/next-check" className="text-sky-800 underline">open</Link>)
                  </>
                )}
              </p>
            )}
          </section>
          <button type="button" onClick={remove} className="rounded-md border border-slate-300 px-3 py-1.5 text-sm text-slate-700 hover:bg-slate-100">
            Delete this case and its data now
          </button>
          {deleteError && <ErrorState error={deleteError} />}
        </>
      )}
      <p className="text-xs text-slate-500" data-testid="copilot-state">
        Technician copilot: {health ? health.copilot : "unknown"}
      </p>
    </div>
  );
}
