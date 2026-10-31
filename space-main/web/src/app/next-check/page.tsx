"use client";

import { CheckCard } from "@/components/findings";
import { ProvenancePanel } from "@/components/provenance";
import { EmptyState } from "@/components/states";
import { DiagnosisRefusal, Items, WorkspaceStates } from "@/components/workspace-states";
import type { NextCheckView } from "@/lib/api/types";
import { useCase } from "@/lib/case-context";
import { useCaseResource } from "@/lib/use-case-resource";

function Checks({ n }: { n: NextCheckView }) {
  return (
    <div className="space-y-4" data-testid="next-checks">
      {n.status === "INSUFFICIENT_EVIDENCE" && <DiagnosisRefusal reason={n.refusal_reason} />}
      {n.status === "NO_FINDING" && (
        <EmptyState title="No check is recommended">No sustained excursion beyond the limits you set was found, so there is nothing to check.</EmptyState>
      )}
      {n.checks.length > 0 && (
        <>
          <p className="text-sm text-slate-700">
            Ranked by Phase 6 rules (relevance, then feasibility, then evidence value). A rank is a work priority, not a confidence in any explanation.
          </p>
          <ol className="space-y-3">
            {n.checks.map((c) => (
              <CheckCard key={c.check_id} c={c} best={c.check_id === n.next_best_check} />
            ))}
          </ol>
        </>
      )}
      <Items title="Limitations" items={n.limitations} />
      <ProvenancePanel p={n.provenance} />
    </div>
  );
}

export default function NextCheckPage() {
  const { api } = useCase();
  const res = useCaseResource((id) => api.getNextCheck(id), 0, true);
  return (
    <div className="space-y-4">
      <h1 className="text-xl font-semibold">Next Best Check</h1>
      <WorkspaceStates href="/next-check" res={res} loading="Loading the ranked checks…">
        {(n) => <Checks n={n} />}
      </WorkspaceStates>
    </div>
  );
}
