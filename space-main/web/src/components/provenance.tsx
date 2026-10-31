import type { ResultProvenance } from "@/lib/api/types";
import { formatTimestamp, shortHash } from "@/lib/format";

import { EvidenceBadge } from "./badges";

/** The one-line label every result carries: what the engine analysed the data as, its Phase 5 scope, and any
 * uploader declaration, kept separate and marked unverified (amendment G). */
export function ProvenanceLine({
  evidence,
  scope,
  declared,
  basis,
}: {
  evidence: ResultProvenance["data_evidence_class"];
  scope: ResultProvenance["validation_scope"];
  declared: ResultProvenance["declared_evidence"];
  basis: ResultProvenance["evidence_basis"];
}) {
  return (
    <p className="flex flex-wrap items-center gap-2 text-xs text-slate-600" data-testid="provenance-line">
      <span>Analysed as</span>
      <EvidenceBadge value={evidence} />
      <span>({basis === "BUNDLED_SAMPLE" ? "bundled ColdSense sample" : "anonymous upload"})</span>
      <span>· Phase 5 scope</span>
      <code className="rounded bg-slate-100 px-1">{scope}</code>
      {declared && (
        <span data-testid="declaration" className="rounded bg-amber-50 px-1 text-amber-900 ring-1 ring-amber-300">
          Uploader declared {declared} — declared by uploader, not verified
        </span>
      )}
    </p>
  );
}

export function ProvenancePanel({ p }: { p: ResultProvenance }) {
  return (
    <section className="space-y-2">
      <ProvenanceLine evidence={p.data_evidence_class} scope={p.validation_scope} declared={p.declared_evidence} basis={p.evidence_basis} />
      <details className="rounded-lg border border-slate-200 bg-white p-4 text-sm" data-testid="provenance">
        <summary className="cursor-pointer font-medium text-slate-800">Evidence and provenance</summary>
        <dl className="mt-3 grid grid-cols-1 gap-x-6 gap-y-2 sm:grid-cols-[max-content_1fr]">
          <dt className="text-slate-500">Analysed as</dt>
          <dd>
            <EvidenceBadge value={p.data_evidence_class} />{" "}
            <span className="text-slate-500">{p.evidence_basis === "BUNDLED_SAMPLE" ? "bundled, labelled ColdSense sample" : "anonymous upload: always UNVALIDATED"}</span>
          </dd>
          <dt className="text-slate-500">Uploader declaration</dt>
          <dd>{p.declared_evidence ? `${p.declared_evidence} — declared by uploader, not verified; it changes no result` : "none"}</dd>
          {p.result_evidence_class && (
            <>
              <dt className="text-slate-500">Result evidence</dt>
              <dd>
                <EvidenceBadge value={p.result_evidence_class} />
              </dd>
            </>
          )}
          <dt className="text-slate-500">Phase 5 scope</dt>
          <dd>
            <code className="text-xs">{p.validation_scope}</code>{" "}
            <span className="text-slate-500">
              ({p.validation_scope_source === "PHASE5_ASSESSMENT" ? "from this result's reliability assessment" : "from the Phase 5 evidence-class table"})
            </span>
          </dd>
          <dt className="text-slate-500">Source file</dt>
          <dd className="break-all">
            {p.source_filename} <span className="text-slate-500">({p.source_size_bytes} bytes, sha256 {shortHash(p.source_sha256)})</span>
          </dd>
          <dt className="text-slate-500">Method</dt>
          <dd>
            {p.method}
            {p.method_version && <span className="text-slate-500"> ({p.method_version})</span>}
          </dd>
          <dt className="text-slate-500">Computed</dt>
          <dd>
            {formatTimestamp(p.computed_at)} by ColdSense {p.coldsense_version}
          </dd>
        </dl>
      </details>
    </section>
  );
}
