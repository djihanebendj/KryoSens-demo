"use client";

import Link from "next/link";
import { useState } from "react";

import { ProvenancePanel } from "@/components/provenance";
import { ErrorState, LoadingState } from "@/components/states";
import { TimelineChart } from "@/components/timeline-chart";
import { WorkspaceStates } from "@/components/workspace-states";
import { ApiFailure } from "@/lib/api/client";
import type { AnalysisView, TimelineView } from "@/lib/api/types";
import { useCase } from "@/lib/case-context";
import { formatMinutes, formatTimestamp, humanize } from "@/lib/format";
import { gateFor } from "@/lib/journey";
import { useCaseResource } from "@/lib/use-case-resource";

const LANES: { title: string; types: string[] }[] = [
  { title: "Compressor (recorded state only; never inferred)", types: ["COMPRESSOR_ON", "COMPRESSOR_OFF", "COMPRESSOR_UNKNOWN"] },
  { title: "Threshold excursions (beyond the limits you set)", types: ["THRESHOLD_EXCURSION", "BRIEF_EXCURSION"] },
  { title: "Temperature behaviour (Phase 3 classification)", types: ["TEMP_RISING", "TEMP_FALLING", "TEMP_STABLE", "TEMP_IRREGULAR", "INSUFFICIENT_DATA"] },
  { title: "Data gaps", types: ["DATA_GAP"] },
];

function toNumber(s: string): number | null | "bad" {
  if (s.trim() === "") return null;
  const n = Number(s);
  return Number.isFinite(n) ? n : "bad";
}

function AnalysisForm({ onDone, current }: { onDone: () => void; current: AnalysisView | null }) {
  const { api, caseId, refresh } = useCase();
  const [upper, setUpper] = useState(current?.upper_threshold_c?.toString() ?? "");
  const [lower, setLower] = useState(current?.lower_threshold_c?.toString() ?? "");
  const [busy, setBusy] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);
  const [error, setError] = useState<ApiFailure | null>(null);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    const u = toNumber(upper);
    const l = toNumber(lower);
    if (u === "bad" || l === "bad") return setFormError("Limits must be numbers in °C, or empty.");
    setFormError(null);
    setError(null);
    setBusy(true);
    try {
      await api.runAnalysis(caseId!, { upper_threshold_c: u, lower_threshold_c: l });
      await refresh();
      onDone();
    } catch (err) {
      setError(err instanceof ApiFailure ? err : new ApiFailure(0, { code: "UNEXPECTED_RESPONSE", message: String(err), retryable: false }));
    } finally {
      setBusy(false);
    }
  }

  return (
    <form onSubmit={submit} className="space-y-3 rounded-lg border border-slate-200 bg-white p-5" aria-busy={busy}>
      <p className="text-sm text-slate-700">
        Analysis = Phase 3 event segmentation and Phase 6 diagnostics against the limits <strong>you</strong> set. ColdSense does not decide what temperature
        is abnormal; without a limit, diagnostics refuse.
      </p>
      <div className="grid gap-3 sm:grid-cols-2">
        <label className="block text-sm">
          Upper limit (°C)
          <input id="upper-threshold" inputMode="decimal" value={upper} onChange={(e) => setUpper(e.target.value)} className="mt-1 block w-full rounded border border-slate-300 px-2 py-1" />
        </label>
        <label className="block text-sm">
          Lower limit (°C, optional)
          <input id="lower-threshold" inputMode="decimal" value={lower} onChange={(e) => setLower(e.target.value)} className="mt-1 block w-full rounded border border-slate-300 px-2 py-1" />
        </label>
      </div>
      {formError && (
        <p role="alert" className="text-sm text-rose-800">
          {formError}
        </p>
      )}
      <button type="submit" disabled={busy} className="rounded-md bg-sky-700 px-4 py-2 text-sm font-medium text-white hover:bg-sky-800 disabled:opacity-50">
        {current ? "Run analysis again" : "Run analysis"}
      </button>
      {busy && <LoadingState label="Segmenting the log and assessing explanations…" />}
      {error && <ErrorState error={error} />}
    </form>
  );
}

function Timeline({ t, a }: { t: TimelineView; a: AnalysisView | null }) {
  return (
    <div className="space-y-4" data-testid="timeline">
      <section className="space-y-2 rounded-lg border border-slate-200 bg-white p-5">
        <h2 className="text-base font-semibold">What did the sensor measure, and where are readings missing?</h2>
        <p className="text-xs text-slate-600">
          {formatTimestamp(t.start)} → {formatTimestamp(t.end)} · unit {t.unit} · time basis: {t.time_basis}
        </p>
        <TimelineChart view={t} />
        <p className="text-xs text-slate-600" data-testid="display-note">
          {t.display.downsampled
            ? `Display only: ${t.display.points_shown} of ${t.display.points_total} readings are drawn (${t.display.method}). Every analysis used all readings.`
            : `All ${t.display.points_total} usable readings are drawn.`}{" "}
          Lines are not smoothed or interpolated; grey bands are data gaps and the line is broken there.
        </p>
        {t.gaps.length > 0 && (
          <ul className="text-xs text-slate-700" data-testid="gaps">
            {t.gaps.map((g) => (
              <li key={g.event_id}>
                Data gap {g.event_id}: {formatTimestamp(g.start)} → {formatTimestamp(g.end)} ({formatMinutes(g.duration_min)}) — no readings; not interpolated. {g.rationale}
              </li>
            ))}
          </ul>
        )}
      </section>
      <section className="space-y-3 rounded-lg border border-slate-200 bg-white p-5">
        <h2 className="text-base font-semibold">When was the compressor recorded on or off, and how did Phase 3 classify each period?</h2>
        {!t.compressor_recorded && <p className="text-sm text-slate-600">No compressor state is recorded in this log; it is never inferred from temperature.</p>}
        {LANES.map((lane) => {
          const ev = t.events.filter((e) => lane.types.includes(e.type));
          if (!ev.length) return null;
          return (
            <details key={lane.title} className="text-sm" open={lane.types.includes("THRESHOLD_EXCURSION") || lane.types.includes("DATA_GAP")}>
              <summary className="cursor-pointer font-medium">
                {lane.title} ({ev.length})
              </summary>
              <table className="mt-2 w-full text-left text-xs">
                <thead className="text-slate-500">
                  <tr>
                    <th className="py-1">Event</th>
                    <th>Type</th>
                    <th>Start → end</th>
                    <th>Duration</th>
                    <th>Why</th>
                  </tr>
                </thead>
                <tbody>
                  {ev.map((e) => (
                    <tr key={e.event_id} className="border-t border-slate-100 align-top">
                      <td className="py-1 font-mono">{e.event_id}</td>
                      <td>{humanize(e.type)}</td>
                      <td>
                        {e.start_censored ? "≤ " : ""}
                        {formatTimestamp(e.start)} → {formatTimestamp(e.end)}
                        {e.end_censored ? " (continues?)" : ""}
                      </td>
                      <td>
                        {e.duration_is_lower_bound ? "≥ " : ""}
                        {formatMinutes(e.duration_min)}
                      </td>
                      <td className="text-slate-600">{e.rationale}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </details>
          );
        })}
      </section>
      {a && a.outcome === "REFUSED" && (
        <section role="status" data-state="refusal" className="rounded-lg border border-amber-300 bg-amber-50 p-4 text-sm text-amber-950">
          Phase 3 refused to segment this log: {a.reasons.join("; ")}
        </section>
      )}
      {a && (
        <p className="text-sm">
          Diagnostics: <code>{a.diagnosis.status}</code>
          {a.diagnosis.refusal_reason && ` — ${a.diagnosis.refusal_reason}`}
          {a.diagnosis.next_best_check && (
            <>
              {" "}
              · next best check <code>{a.diagnosis.next_best_check}</code> (<Link href="/next-check" className="text-sky-800 underline">Next Best Check</Link>)
            </>
          )}
        </p>
      )}
      {t.limitations.length > 0 && (
        <details className="text-sm">
          <summary className="cursor-pointer">Limitations ({t.limitations.length})</summary>
          <ul className="mt-1 list-disc pl-5 text-slate-700">
            {t.limitations.map((x, i) => (
              <li key={i}>{x}</li>
            ))}
          </ul>
        </details>
      )}
      <ProvenancePanel p={t.provenance} />
    </div>
  );
}

export default function TimelinePage() {
  const { api, status } = useCase();
  const [version, setVersion] = useState(0);
  const res = useCaseResource(async (id) => {
    const [a, t] = await Promise.all([api.getAnalysis(id), api.getTimeline(id)]);
    return { a, t };
  }, version, true);
  const gate = gateFor("/timeline", status);
  const notRun = gate.enabled && (res.state === "not-run" || (res.state === "error" && res.error.code === "ANALYSIS_NOT_RUN"));
  return (
    <div className="space-y-4">
      <h1 className="text-xl font-semibold">Temperature Timeline</h1>
      {gate.enabled && (notRun || res.state === "ready") && (
        <AnalysisForm current={res.state === "ready" ? res.data.a : null} onDone={() => setVersion((v) => v + 1)} />
      )}
      {notRun ? (
        <p className="text-sm text-slate-600" data-state="analysis-not-run">
          No analysis yet: set a limit above and run it to see the readings and events.
        </p>
      ) : (
        <WorkspaceStates href="/timeline" res={res} loading="Loading the timeline…">
          {({ a, t }) => <Timeline t={t} a={a} />}
        </WorkspaceStates>
      )}
    </div>
  );
}
