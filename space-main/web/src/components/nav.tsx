"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

import { useCase } from "@/lib/case-context";
import { gateFor, journey } from "@/lib/journey";

import { EvidenceBadge } from "./badges";

// The 11 sections of spec §8, in its order. `available: false` = not built yet (a later stage).
export const PAGES = [
  { href: "/", label: "Home", available: true },
  { href: "/dashboard", label: "Dashboard", available: true },
  { href: "/upload", label: "Upload Data", available: true },
  { href: "/quality", label: "Data Quality", available: true },
  { href: "/timeline", label: "Temperature Timeline", available: true },
  { href: "/prediction", label: "Prediction", available: true },
  { href: "/evidence", label: "Evidence", available: true },
  { href: "/next-check", label: "Next Best Check", available: true },
  { href: "/copilot", label: "Technician Copilot", available: true },
  { href: "/investigation", label: "Investigation", available: true },
  { href: "/incident", label: "Incident", available: true },
  { href: "/reports", label: "Reports", available: true },
] as const;

export function Nav() {
  const path = usePathname();
  const { status } = useCase();
  return (
    <nav aria-label="Main" className="flex flex-col gap-4">
      <ul className="flex flex-wrap gap-1 md:flex-col">
        {PAGES.map((p) => {
          const active = p.href === "/" ? path === "/" : path.startsWith(p.href);
          const gate = gateFor(p.href, status);
          const base = "flex items-center justify-between gap-2 rounded-md px-3 py-1.5 text-sm";
          if (!gate.enabled)
            return (
              <li key={p.href}>
                <span aria-disabled="true" title={gate.reason} data-testid={`nav-disabled-${p.href}`} className={`${base} cursor-not-allowed text-slate-400`}>
                  {p.label}
                  <span className="sr-only"> (unavailable: {gate.reason})</span>
                </span>
              </li>
            );
          return (
            <li key={p.href}>
              <Link
                href={p.href}
                aria-current={active ? "page" : undefined}
                className={`${base} ${active ? "bg-sky-100 font-medium text-sky-900" : "text-slate-700 hover:bg-slate-100"} ${p.available ? "" : "text-slate-400"}`}
              >
                {p.label}
                {!p.available && <span className="font-mono text-[10px] text-slate-400">N/A</span>}
              </Link>
            </li>
          );
        })}
      </ul>
      {status && (
        <div className="hidden rounded-md border border-slate-200 bg-white p-3 text-xs md:block" data-testid="current-case">
          <p className="text-slate-500">Current case</p>
          <p className="mt-1 break-all font-medium text-slate-800">{status.filename}</p>
          <p className="mt-1">
            <EvidenceBadge value={status.evidence_class} label="analysed as" />
          </p>
          {status.declared_evidence && <p className="mt-1 text-amber-800">declared {status.declared_evidence} by uploader, not verified</p>}
        </div>
      )}
    </nav>
  );
}

const STEP_STYLE = {
  done: "border-emerald-300 bg-emerald-50 text-emerald-900",
  refused: "border-amber-300 bg-amber-50 text-amber-900",
  available: "border-sky-300 bg-white text-sky-900",
  blocked: "border-slate-200 bg-slate-50 text-slate-400",
  "not-available": "border-dashed border-slate-200 bg-white text-slate-400",
};
const STEP_WORD = { done: "done", refused: "refused", available: "next", blocked: "blocked", "not-available": "not available" };

/** Workflow progress (spec §8 main journey): what is done, what is next, what is blocked and why. */
export function JourneyBar() {
  const { status } = useCase();
  const steps = journey(status);
  return (
    <ol aria-label="Workflow progress" className="flex flex-wrap gap-1.5 text-xs" data-testid="journey">
      {steps.map((s, i) => {
        const inner = (
          <>
            <span className="font-mono">{i + 1}.</span> {s.label}{" "}
            <span className="opacity-75">
              ({STEP_WORD[s.state]}
              {s.note ? `: ${s.note}` : ""})
            </span>
          </>
        );
        return (
          <li key={s.label} data-step-state={s.state} className={`rounded border px-2 py-1 ${STEP_STYLE[s.state]}`}>
            {s.href && (s.state === "available" || s.state === "done" || s.state === "refused") ? (
              <Link href={s.href} className="hover:underline">
                {inner}
              </Link>
            ) : (
              inner
            )}
          </li>
        );
      })}
    </ol>
  );
}
