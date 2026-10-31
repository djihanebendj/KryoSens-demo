"use client";

import Link from "next/link";

import { EvidenceBadge, VerdictBadge } from "@/components/badges";
import { ProvenancePanel } from "@/components/provenance";
import { EmptyState, ErrorState, LoadingState } from "@/components/states";
import type { QualityView } from "@/lib/api/types";
import { useCase } from "@/lib/case-context";
import { formatNumber, formatTimestamp } from "@/lib/format";
import { useCaseResource } from "@/lib/use-case-resource";

const SEVERITY_TONE: Record<string, string> = {
  BLOCKING: "border-rose-300 bg-rose-50 text-rose-900",
  WARNING: "border-amber-300 bg-amber-50 text-amber-900",
  INFO: "border-slate-200 bg-slate-50 text-slate-700",
};

function Quality({ q }: { q: QualityView }) {
  const h = q.headline;
  const r = q.report;
  return (
    <div className="space-y-4" data-testid="quality">
      <section className="rounded-lg border border-slate-200 bg-white p-5">
        <div className="flex flex-wrap items-center gap-2">
          <h2 className="text-lg font-semibold">Verdict</h2>
          <VerdictBadge value={h.verdict} />
          <EvidenceBadge value={r.evidence} label="data" />
        </div>
        <p className="mt-2 text-sm text-slate-700" data-testid="suitability">
          {h.suitable_for_analysis
            ? "The data can be used for analysis" + (h.verdict === "LIMITED" ? ", with the warnings below." : ".")
            : "The data is not suitable for analysis. Later steps are blocked until a better log is uploaded."}
        </p>
        <dl className="mt-4 grid grid-cols-2 gap-3 text-sm sm:grid-cols-4">
          {[
            ["Rows read", String(h.rows_read)],
            ["Rows usable", String(h.rows_usable)],
            ["Duration", `${formatNumber(h.duration_hours, 1)} h`],
            ["Coverage", `${formatNumber(h.coverage_pct, 1)} %`],
            ["Start", formatTimestamp(h.start)],
            ["End", formatTimestamp(h.end)],
            ["Median interval", r.median_interval_s == null ? "—" : `${formatNumber(r.median_interval_s, 0)} s`],
            ["Gaps", String(r.n_gaps)],
          ].map(([k, v]) => (
            <div key={k} className="rounded border border-slate-100 bg-slate-50 p-2">
              <dt className="text-xs text-slate-500">{k}</dt>
              <dd className="font-medium">{v}</dd>
            </div>
          ))}
        </dl>
        {r.input && <p className="mt-3 text-xs text-slate-500">Time handling: {r.input.timezone_handling}</p>}
      </section>
      <section className="space-y-2">
        <h2 className="text-base font-semibold">Issues ({q.issues.length})</h2>
        {q.issues.length === 0 && <p className="text-sm text-slate-600">No issues were found.</p>}
        <ul className="space-y-2">
          {q.issues.map((i, k) => (
            <li key={k} className={`rounded border p-2 text-sm ${SEVERITY_TONE[i.severity] ?? SEVERITY_TONE.INFO}`}>
              <span className="font-mono text-xs">{i.severity}</span> <code className="text-xs">{i.code}</code> — {i.message}
            </li>
          ))}
        </ul>
      </section>
      {q.limitations.length > 0 && (
        <section className="text-sm">
          <h2 className="text-base font-semibold">Limitations</h2>
          <ul className="mt-1 list-disc pl-5 text-slate-700">
            {q.limitations.map((x, i) => (
              <li key={i}>{x}</li>
            ))}
          </ul>
        </section>
      )}
      <ProvenancePanel p={q.provenance} />
      {h.suitable_for_analysis && (
        <Link href="/prediction" className="inline-block rounded-md bg-sky-700 px-4 py-2 text-sm font-medium text-white hover:bg-sky-800">
          Continue to Prediction
        </Link>
      )}
    </div>
  );
}

export default function QualityPage() {
  const { api } = useCase();
  const res = useCaseResource((id) => api.getQuality(id));
  return (
    <div className="space-y-4">
      <h1 className="text-xl font-semibold">Data Quality</h1>
      {res.state === "no-case" && (
        <EmptyState title="No data uploaded yet" action={{ href: "/upload", label: "Upload Data" }}>
          Upload a log to see its data-quality report.
        </EmptyState>
      )}
      {res.state === "loading" && <LoadingState label="Loading the data-quality report…" />}
      {res.state === "error" && <ErrorState error={res.error} onRetry={res.retry} />}
      {res.state === "ready" && <Quality q={res.data} />}
    </div>
  );
}
