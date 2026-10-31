"use client";

import Link from "next/link";

import type { ApiFailure } from "@/lib/api/client";
import { useCase } from "@/lib/case-context";
import { gateFor } from "@/lib/journey";
import type { Resource } from "@/lib/use-case-resource";

import { EmptyState, ErrorState, LoadingState } from "./states";

/** The shared empty / blocked / loading / error / analysis-not-run states of an analysis-workspace page. */
export function WorkspaceStates<T>({
  href,
  res,
  loading,
  children,
}: {
  href: string;
  res: Resource<T>;
  loading: string;
  children: (data: T) => React.ReactNode;
}) {
  const { status } = useCase();
  if (res.state === "no-case")
    return (
      <EmptyState title="No data uploaded yet" action={{ href: "/upload", label: "Upload Data" }}>
        Upload a log or use a bundled sample to start a case.
      </EmptyState>
    );
  if (res.state === "loading") return <LoadingState label={loading} />;
  const gate = gateFor(href, status);
  if (status && !gate.enabled && gate.reason.startsWith("the data-quality"))
    return (
      <section role="status" data-state="blocked" className="rounded-lg border border-rose-300 bg-rose-50 p-5 text-sm text-rose-900">
        Blocked: {gate.reason}. See <Link href="/quality" className="underline">Data Quality</Link> for the reasons.
      </section>
    );
  if (res.state === "not-run") return <AnalysisNeeded />;
  if (res.state === "error") return res.error.code === "ANALYSIS_NOT_RUN" ? <AnalysisNeeded /> : <ErrorState error={res.error as ApiFailure} onRetry={res.retry} />;
  return <>{children(res.data)}</>;
}

export function AnalysisNeeded() {
  return (
    <section role="status" data-state="analysis-not-run" className="rounded-lg border border-dashed border-slate-300 bg-white p-6 text-sm">
      <h2 className="font-semibold text-slate-800">The analysis has not been run for this case</h2>
      <p className="mt-1 text-slate-600">
        Set the temperature limit(s) to analyse against on Temperature Timeline. ColdSense does not decide what temperature is abnormal.
      </p>
      <Link href="/timeline" className="mt-3 inline-block rounded-md bg-sky-700 px-4 py-2 text-white hover:bg-sky-800">
        Go to Temperature Timeline
      </Link>
    </section>
  );
}

/** Phase 6 refused to assess (INSUFFICIENT_EVIDENCE): its reason, never invented content. */
export function DiagnosisRefusal({ reason }: { reason: string | null }) {
  return (
    <section role="status" data-state="refusal" className="rounded-lg border border-amber-300 bg-amber-50 p-5 text-sm text-amber-950">
      <h2 className="font-semibold">INSUFFICIENT_EVIDENCE — ColdSense did not assess any explanation</h2>
      <p className="mt-1">{reason ?? "no reason was given"}</p>
      <p className="mt-2 text-xs text-amber-800">A refusal is a valid result: nothing is proposed, ranked or checked without evidence.</p>
    </section>
  );
}

export function Label({ kind }: { kind: "OBSERVED_FACT" | "MODEL_INFERENCE" | "SYSTEM_RECOMMENDATION" | "DOCUMENT_FACT" }) {
  const tone = {
    OBSERVED_FACT: "bg-slate-100 text-slate-800 ring-slate-300",
    MODEL_INFERENCE: "bg-violet-100 text-violet-900 ring-violet-300",
    SYSTEM_RECOMMENDATION: "bg-sky-100 text-sky-900 ring-sky-300",
    DOCUMENT_FACT: "bg-emerald-100 text-emerald-900 ring-emerald-300",
  }[kind];
  return <span className={`inline-block rounded px-1.5 py-0.5 font-mono text-[11px] ring-1 ${tone}`}>{kind}</span>;
}

export function Items({ title, items = [], tone, testId }: { title: string; items?: string[]; tone?: string; testId?: string }) {
  return (
    <div data-testid={testId}>
      <p className={`text-xs font-semibold ${tone ?? "text-slate-600"}`}>
        {title} ({items.length})
      </p>
      {items.length ? (
        <ul className="mt-1 list-disc space-y-0.5 pl-5">
          {items.map((x, i) => (
            <li key={i}>{x}</li>
          ))}
        </ul>
      ) : (
        <p className="text-xs text-slate-400">none recorded</p>
      )}
    </div>
  );
}
