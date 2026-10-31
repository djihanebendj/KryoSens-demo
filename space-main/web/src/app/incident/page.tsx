"use client";

// Incident: the finding entry form plus the confirmed records and drafts. `useCaseResource` keeps the last incident
// view visible while a same-case refresh is in flight, so preparing or confirming a draft does not unmount the
// exact-draft review panel and lose its single-use token. A different case never shows the previous case's data.
import { useState } from "react";

import { FindingsPanel } from "@/components/findings-panel";
import { ProvenancePanel } from "@/components/provenance";
import { EmptyState, ErrorState, LoadingState } from "@/components/states";
import { useCase } from "@/lib/case-context";
import { useCaseResource } from "@/lib/use-case-resource";

export default function IncidentPage() {
  const { caseId, api } = useCase();
  const [reloadKey, setReloadKey] = useState(0);
  const res = useCaseResource((id) => api.getIncident(id), reloadKey);

  if (!caseId || res.state === "no-case")
    return (
      <EmptyState title="No data uploaded yet" action={{ href: "/upload", label: "Upload Data" }}>
        Upload a log or use a bundled sample, then record and confirm technician findings here.
      </EmptyState>
    );
  if (res.state === "error") return <ErrorState error={res.error} onRetry={res.retry} />;
  if (res.state === "loading" || res.state === "not-run")
    return <LoadingState label="Loading incident records…" />;

  return (
    <div className="space-y-4">
      <header className="space-y-1">
        <h1 className="text-2xl font-semibold tracking-tight">Incident</h1>
        <p className="text-sm text-slate-600">
          Technician findings, drafts and outcomes. A finding is your statement, not a model inference; only an
          explicit confirmation writes a record. The report builder never treats a draft as history.
        </p>
      </header>
      <div className="space-y-6">
        <ProvenancePanel p={res.data.provenance} />
        <FindingsPanel caseId={caseId} incident={res.data} onChanged={() => setReloadKey((n) => n + 1)} />
      </div>
    </div>
  );
}
