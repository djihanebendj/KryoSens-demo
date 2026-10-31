"use client";

// Guided Start: a rule-based, read-only view of what the backend already decided for this case. Every dynamic value
// comes from the existing GET views (next-check, investigation, evidence); nothing is computed, ranked or generated
// here, there is no free-text input, and the copilot endpoint is never called.
import { ProvenancePanel } from "@/components/provenance";
import { EmptyState } from "@/components/states";
import { DiagnosisRefusal, Items, Label, WorkspaceStates } from "@/components/workspace-states";
import type { EvidenceView, InvestigationView, NextCheckView } from "@/lib/api/types";
import { useCase } from "@/lib/case-context";
import { useCaseResource } from "@/lib/use-case-resource";

export const FREE_TEXT_NOTICE =
  "Free-text assistant: disabled in this build. No live language model is connected, and the assistant never " +
  "confirms faults. Guided Start: rule-based, no AI text generation.";
export const SAFETY_NOTICE =
  "Work on refrigerant circuits or live electrical equipment only by qualified personnel. This tool gives no repair " +
  "procedures.";

/** Plain-language labels for known backend enum codes. An unknown code is shown as it is (raw fallback). */
export const CODE_LABELS: Record<string, string> = {
  REVIEW_OPERATIONAL_RECORDS: "Review the site's operational records",
  CORROBORATE_RECORDS: "Corroborate the existing records",
  RECOVER_MISSING_READINGS: "Recover the missing readings",
  REFERENCE_READING: "Take a reference temperature reading",
  PROPOSED: "Proposed hypothesis",
  UNDER_TEST: "Hypothesis under test",
  INCONCLUSIVE: "Inconclusive hypothesis",
  REJECTED: "Rejected hypothesis",
  SUPPORTED: "Supported hypothesis",
  EXISTING_RECORDS: "existing records",
  SPOT_CHECK: "spot check",
  SITE_INSPECTION: "site inspection",
  OFFICE_ONLY: "office only",
  ROOM_ENTRY: "room entry",
  EQUIPMENT_ACCESS: "equipment access",
};

export function labelFor(code: string | null | undefined): string {
  if (!code) return "";
  return CODE_LABELS[code] ?? code;
}

type Views = { next: NextCheckView; inv: InvestigationView; ev: EvidenceView };

function Refusals({ v }: { v: Views }) {
  const diagnosisRefused = v.inv.status === "INSUFFICIENT_EVIDENCE";
  const predictionRefused = v.ev.prediction?.outcome === "REFUSED";
  if (!diagnosisRefused && !predictionRefused) return null;
  return (
    <div className="space-y-3" data-testid="guided-refusal">
      {diagnosisRefused && <DiagnosisRefusal reason={v.inv.refusal_reason} />}
      {predictionRefused && (
        <section role="status" data-state="refusal" className="rounded-lg border border-amber-300 bg-amber-50 p-5 text-sm text-amber-950">
          <h2 className="font-semibold">Prediction refused</h2>
          <p className="mt-1">{v.ev.prediction?.refusal_reason ?? "no reason was given"}</p>
        </section>
      )}
    </div>
  );
}

function Panels({ v }: { v: Views }) {
  const top = v.next.checks.find((c) => c.check_id === v.next.next_best_check) ?? null;
  return (
    <div className="space-y-5">
      <section className="space-y-2 rounded-lg border border-slate-200 bg-white p-5 text-sm" data-testid="where-to-start">
        <h2 className="text-base font-semibold">Where to start</h2>
        {top ? (
          <>
            <p className="flex flex-wrap items-center gap-2">
              <Label kind={v.next.label} />
              <span className="font-medium">{labelFor(top.check_id)}</span>
              <code className="text-xs text-slate-500">{top.check_id}</code>
            </p>
            <p>{top.action}</p>
            <p className="text-slate-700">
              <span className="font-medium">Why: </span>
              {top.why_useful}
            </p>
            <Items title="What it confirms or excludes" items={top.expected_information} testId="expected-information" />
            <p className="text-xs text-slate-600">
              Effort: {labelFor(top.effort)} · Access: {labelFor(top.safety)}
            </p>
          </>
        ) : (
          <EmptyState title="No ranked check">The backend returned no next check for this case.</EmptyState>
        )}
      </section>

      <section className="space-y-3" data-testid="leading-hypotheses">
        <h2 className="text-base font-semibold">Leading hypotheses</h2>
        {v.inv.hypotheses.length === 0 && (
          <EmptyState title="No hypothesis">The backend proposed no hypothesis for this case.</EmptyState>
        )}
        {v.inv.hypotheses.map((h) => (
          <article key={h.id} className="space-y-2 rounded-lg border border-violet-200 bg-white p-4 text-sm" data-testid={`guided-hypothesis-${h.id}`}>
            <header className="flex flex-wrap items-center gap-2">
              <Label kind={v.inv.hypothesis_label} />
              <span className="rounded bg-violet-50 px-1.5 text-xs text-violet-900 ring-1 ring-violet-200">{v.inv.hypothesis_note}</span>
              <span className="text-xs">{labelFor(h.status)}</span>
              <span className="font-mono text-xs text-slate-500">{h.id}</span>
            </header>
            <p className="font-medium">{h.statement}</p>
            <div className="grid gap-2 sm:grid-cols-2">
              <Items title="Supporting evidence" items={h.supporting_evidence} tone="text-emerald-800" />
              <Items title="Contradicting evidence" items={h.contradicting_evidence} tone="text-rose-800" />
            </div>
          </article>
        ))}
      </section>

      <section className="space-y-2" data-testid="data-still-needed">
        <h2 className="text-base font-semibold">Data you still need</h2>
        {v.inv.hypotheses.every((h) => !h.data_needed?.length) ? (
          <p className="text-sm text-slate-600">The backend listed no missing evidence.</p>
        ) : (
          v.inv.hypotheses
            .filter((h) => h.data_needed?.length)
            .map((h) => <Items key={h.id} title={`For ${h.id}`} items={h.data_needed} tone="text-amber-800" />)
        )}
      </section>
    </div>
  );
}

function GuidedContent({ v }: { v: Views }) {
  const refused = v.inv.status === "INSUFFICIENT_EVIDENCE" || v.ev.prediction?.outcome === "REFUSED";
  return (
    <div className="space-y-5" data-testid="guided-start">
      <ProvenancePanel p={v.inv.provenance} />
      {refused ? <Refusals v={v} /> : <Panels v={v} />}
    </div>
  );
}

export function GuidedStart() {
  const { api } = useCase();
  const res = useCaseResource<Views>(
    async (id) => {
      const [next, inv, ev] = await Promise.all([api.getNextCheck(id), api.getInvestigation(id), api.getEvidence(id)]);
      return { next, inv, ev };
    },
    0,
    true,
  );
  return (
    <section className="space-y-4" aria-labelledby="guided-start-title">
      <h2 id="guided-start-title" className="text-xl font-semibold">Guided Start</h2>
      <WorkspaceStates href="/investigation" res={res} loading="Loading the backend's next check and hypotheses…">
        {(v) => <GuidedContent v={v} />}
      </WorkspaceStates>
    </section>
  );
}
