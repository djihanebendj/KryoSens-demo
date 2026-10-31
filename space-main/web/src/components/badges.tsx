import type { EvidenceClass, StepState, Verdict } from "@/lib/api/types";

const tone = {
  good: "bg-emerald-100 text-emerald-900 ring-emerald-300",
  warn: "bg-amber-100 text-amber-900 ring-amber-300",
  bad: "bg-rose-100 text-rose-900 ring-rose-300",
  neutral: "bg-slate-100 text-slate-800 ring-slate-300",
  info: "bg-sky-100 text-sky-900 ring-sky-300",
};

function Badge({ t, children, title }: { t: keyof typeof tone; children: React.ReactNode; title?: string }) {
  return (
    <span title={title} className={`inline-flex items-center rounded px-2 py-0.5 font-mono text-xs font-medium ring-1 ${tone[t]}`}>
      {children}
    </span>
  );
}

const EVIDENCE_TITLE: Record<EvidenceClass, string> = {
  REAL: "measured on a physical unit (declared by the uploader)",
  SIMULATED: "produced by a simulator: never evidence about a real room",
  EXPECTED: "a model output not yet checked against a real outcome",
  UNVALIDATED: "provenance unknown: treated neither as measured nor as simulated",
};

export function EvidenceBadge({ value, label }: { value: EvidenceClass; label?: string }) {
  return (
    <Badge t={value === "SIMULATED" ? "info" : value === "REAL" ? "good" : "neutral"} title={EVIDENCE_TITLE[value]}>
      {label ? `${label}: ` : ""}
      {value}
    </Badge>
  );
}

export function VerdictBadge({ value }: { value: Verdict }) {
  return <Badge t={value === "USABLE" ? "good" : value === "LIMITED" ? "warn" : "bad"}>{value}</Badge>;
}

export function StepBadge({ value }: { value: StepState }) {
  const t = value === "READY" ? "good" : value === "REFUSED" || value === "BLOCKED" ? "bad" : "neutral";
  return <Badge t={t}>{value}</Badge>;
}

export function StatusBadge({ value, t }: { value: string; t: keyof typeof tone }) {
  return <Badge t={t}>{value}</Badge>;
}
