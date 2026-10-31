"use client";

import Link from "next/link";

import { CheckCard, FindingCard } from "@/components/findings";
import { KnowledgePanel } from "@/components/knowledge-panel";
import { ProvenancePanel } from "@/components/provenance";
import { DiagnosisRefusal, Items, Label, WorkspaceStates } from "@/components/workspace-states";
import type { EvidenceView } from "@/lib/api/types";
import { useCase } from "@/lib/case-context";
import { formatTimestamp } from "@/lib/format";
import { useCaseResource } from "@/lib/use-case-resource";

function Support({ e }: { e: EvidenceView }) {
  const p = e.prediction;
  return (
    <section className="rounded-lg border border-slate-200 bg-white p-4 text-sm" data-testid="support">
      <h2 className="font-semibold">Support and refusal status</h2>
      <dl className="mt-2 grid grid-cols-1 gap-x-6 gap-y-1 sm:grid-cols-[max-content_1fr]">
        <dt className="text-slate-500">Data quality</dt>
        <dd>{e.data_verdict}</dd>
        <dt className="text-slate-500">Diagnostics (Phase 6)</dt>
        <dd>
          <code>{e.diagnosis_status}</code>
          {e.diagnosis_refusal_reason && ` — ${e.diagnosis_refusal_reason}`}
          <span className="block text-xs text-slate-500">Rules: {e.rules_validation}</span>
        </dd>
        <dt className="text-slate-500">Prediction</dt>
        <dd data-testid="prediction-status">
          {p.outcome === "NOT_RUN" && (
            <>
              not requested (<Link href="/prediction" className="text-sky-800 underline">Prediction</Link>)
            </>
          )}
          {p.outcome === "REFUSED" && (
            <>
              refused: <code>{p.refusal_code}</code> — {p.refusal_reason}
            </>
          )}
          {p.outcome === "FORECAST" && (
            <>
              forecast available; reliability <code>{p.reliability_status ?? "not assessed"}</code> (rule-based, not a probability), scope{" "}
              <code>{p.validation_scope}</code>
            </>
          )}
        </dd>
      </dl>
    </section>
  );
}

function Evidence({ e }: { e: EvidenceView }) {
  const contradictions = e.findings.flatMap((f) => f.explanations.filter((x) => (x.contradicting ?? []).length).map((x) => ({ f, x })));
  const missing = e.findings.flatMap((f) => f.explanations.filter((x) => (x.missing ?? []).length).map((x) => ({ f, x })));
  const knowledgeQuery = e.recommendations[0]?.action ?? e.findings[0]?.statement ?? "";
  return (
    <div className="space-y-5" data-testid="evidence">
      <Support e={e} />
      {e.diagnosis_status === "INSUFFICIENT_EVIDENCE" && <DiagnosisRefusal reason={e.diagnosis_refusal_reason} />}
      <section className="space-y-2">
        <h2 className="text-base font-semibold">
          <Label kind="OBSERVED_FACT" /> Observed facts from the log ({e.observed_facts.length})
        </h2>
        <ul className="space-y-1 text-sm">
          {e.observed_facts.map((f) => (
            <li key={f.fact_id} className="rounded border border-slate-200 bg-white p-2">
              <span className="font-mono text-xs">{f.fact_id}</span> <span className="text-xs text-slate-500">{f.source} · {f.kind} · {f.evidence}</span>
              <p>{f.statement}</p>
              <p className="text-xs text-slate-500">
                {f.start && `${formatTimestamp(f.start)} → ${formatTimestamp(f.end)}`}
                {(f.source_lines ?? []).length > 0 && ` · source lines ${Math.min(...(f.source_lines ?? []))}–${Math.max(...(f.source_lines ?? []))}`}
                {(f.source_ids ?? []).length > 0 && ` · ${(f.source_ids ?? []).join(", ")}`}
              </p>
            </li>
          ))}
        </ul>
      </section>
      <section className="rounded-lg border border-dashed border-slate-300 bg-white p-4 text-sm" data-state="not-available" data-testid="document-facts">
        <h2 className="font-semibold">
          <Label kind="DOCUMENT_FACT" /> Automatic documentation facts: <code>{e.document_facts.status}</code>
        </h2>
        <p className="text-slate-600">{e.document_facts.reason}</p>
        <p className="text-slate-600">
          The diagnosis does not infer document facts from this case. Use the read-only documentation search below
          to retrieve cited passages from the local Phase 7 index; the result is quoted text, not a diagnosis.
        </p>
      </section>
      <KnowledgePanel key={knowledgeQuery} defaultQuery={knowledgeQuery} />
      <section className="space-y-2">
        <h2 className="text-base font-semibold">
          <Label kind="MODEL_INFERENCE" /> Rule-based assessments (hypotheses, not confirmed faults)
        </h2>
        {e.findings.length === 0 && e.diagnosis_status === "NO_FINDING" && (
          <p className="text-sm text-slate-600">No sustained excursion beyond the limits you set, so no explanation was assessed.</p>
        )}
        <div className="grid gap-3 sm:grid-cols-2" data-testid="evidence-summary">
          <div className="rounded border border-rose-200 bg-rose-50 p-3 text-sm">
            <p className="font-semibold text-rose-900">Contradictions ({contradictions.length})</p>
            <ul className="list-disc pl-5">
              {contradictions.map(({ f, x }) => (
                <li key={`${f.finding_id}-${x.code}`}>
                  {f.finding_id} {x.code}: {(x.contradicting ?? []).join("; ")}
                </li>
              ))}
            </ul>
          </div>
          <div className="rounded border border-amber-200 bg-amber-50 p-3 text-sm">
            <p className="font-semibold text-amber-900">Missing evidence ({missing.length})</p>
            <ul className="list-disc pl-5">
              {missing.map(({ f, x }) => (
                <li key={`${f.finding_id}-${x.code}`}>
                  {f.finding_id} {x.code}: {(x.missing ?? []).join("; ")}
                </li>
              ))}
            </ul>
          </div>
        </div>
        {e.findings.map((f) => (
          <FindingCard key={f.finding_id} f={f} />
        ))}
      </section>
      {e.recommendations.length > 0 && (
        <section className="space-y-2">
          <h2 className="text-base font-semibold">
            <Label kind="SYSTEM_RECOMMENDATION" /> Recommended checks ({e.recommendations.length}) — <Link href="/next-check" className="text-sky-800 underline">Next Best Check</Link>
          </h2>
          <ol className="space-y-2">
            {e.recommendations.slice(0, 1).map((c) => (
              <CheckCard key={c.check_id} c={c} best />
            ))}
          </ol>
        </section>
      )}
      {(e.data_quality_limitations.length > 0 || e.limitations.length > 0) && (
        <section className="grid gap-3 text-sm sm:grid-cols-2">
          <Items title="Data-quality limitations" items={e.data_quality_limitations} />
          <Items title="Limitations" items={e.limitations} />
        </section>
      )}
      <ProvenancePanel p={e.provenance} />
    </div>
  );
}

export default function EvidencePage() {
  const { api } = useCase();
  const res = useCaseResource((id) => api.getEvidence(id), 0, true);
  return (
    <div className="space-y-4">
      <h1 className="text-xl font-semibold">Evidence</h1>
      <WorkspaceStates href="/evidence" res={res} loading="Loading the evidence…">
        {(e) => <Evidence e={e} />}
      </WorkspaceStates>
    </div>
  );
}
