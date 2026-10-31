import type { Finding, NextCheck } from "@/lib/api/types";
import { formatTimestamp, humanize } from "@/lib/format";

import { Items, Label } from "./workspace-states";

/** A Phase 6 finding and its explanation assessments: supporting, contradicting, neutral and missing evidence,
 * exactly as returned. Every assessment is a MODEL_INFERENCE, never a confirmed cause. */
export function FindingCard({ f }: { f: Finding }) {
  return (
    <article className="space-y-3 rounded-lg border border-slate-200 bg-white p-4 text-sm" data-testid={`finding-${f.finding_id}`}>
      <header className="flex flex-wrap items-center gap-2">
        <h3 className="font-semibold">
          {f.finding_id}: excursion {f.side} {f.limit_c} °C
        </h3>
        <code className="text-xs">{f.status}</code>
        <span className="text-xs text-slate-500">
          window {formatTimestamp(f.window_start)} → {formatTimestamp(f.window_end)} · fact {f.fact_id} · event {f.event_id}
        </span>
      </header>
      <p className="text-slate-700">
        <Label kind="OBSERVED_FACT" /> {f.statement}
      </p>
      <ul className="space-y-2">
        {f.explanations.map((x) => (
          <li key={x.code} className="rounded border border-slate-100 bg-slate-50 p-3" data-testid={`explanation-${f.finding_id}-${x.code}`}>
            <p>
              <Label kind="MODEL_INFERENCE" /> <strong>{humanize(x.code)}</strong>: {x.statement}
            </p>
            <div className="mt-2 grid gap-2 sm:grid-cols-2">
              <Items title="Supporting" items={x.supporting} tone="text-emerald-800" />
              <Items title="Contradicting" items={x.contradicting} tone="text-rose-800" testId="contradicting" />
              <Items title="Missing evidence" items={x.missing} tone="text-amber-800" testId="missing" />
              <Items title="Neutral" items={x.neutral} />
            </div>
            {x.required_next_check && (
              <p className="mt-2 text-xs">
                <Label kind="SYSTEM_RECOMMENDATION" /> next check for this explanation: <code>{x.required_next_check}</code>
              </p>
            )}
          </li>
        ))}
      </ul>
      {(f.limitations ?? []).length > 0 && <Items title="Limitations" items={f.limitations} />}
    </article>
  );
}

/** One Phase 6 check, with its rank, reasons and expected information as returned. No score is shown or derived. */
export function CheckCard({ c, best }: { c: NextCheck; best: boolean }) {
  return (
    <li className={`space-y-2 rounded-lg border bg-white p-4 text-sm ${best ? "border-sky-400" : "border-slate-200"}`} data-testid={`check-${c.check_id}`}>
      <header className="flex flex-wrap items-center gap-2">
        <span className="font-mono text-xs">rank {c.rank}</span>
        <Label kind="SYSTEM_RECOMMENDATION" />
        <h3 className="font-semibold">{c.check_id}</h3>
        {best && <span className="rounded bg-sky-100 px-1.5 text-xs text-sky-900">NEXT BEST CHECK</span>}
      </header>
      <p>{c.action}</p>
      <dl className="grid grid-cols-1 gap-x-4 gap-y-1 text-xs sm:grid-cols-[max-content_1fr]">
        <dt className="text-slate-500">Why</dt>
        <dd>{c.why_useful}</dd>
        <dt className="text-slate-500">Needs</dt>
        <dd>
          {c.required_input}
          {c.units && ` [${c.units}]`}
          {c.resolution_or_duration && `, ${c.resolution_or_duration}`}
        </dd>
        <dt className="text-slate-500">Windows</dt>
        <dd>{c.windows.join("; ")}</dd>
        <dt className="text-slate-500">Resolves</dt>
        <dd>
          {c.resolves.join(", ")} ({c.open_items} open item(s): a count, not a score)
        </dd>
        <dt className="text-slate-500">Effort / evidence value / safety</dt>
        <dd>
          {c.effort} / {c.evidence_value} / {c.safety}
        </dd>
        <dt className="text-slate-500">Ranking basis</dt>
        <dd className="text-slate-600">{c.ranking_basis}</dd>
      </dl>
      <div className="grid gap-2 sm:grid-cols-2">
        <Items title="Evidence" items={c.evidence} />
        <Items title="Expected information" items={c.expected_information} testId="expected-information" />
      </div>
      <p className="text-xs text-amber-900">Safety: {c.safety_note}</p>
    </li>
  );
}
