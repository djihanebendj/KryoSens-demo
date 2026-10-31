import type { PredictionRefusal, PredictionSuccess, RecommendationView } from "@/lib/api/types";
import { formatMinutes, formatTemp, formatTimestamp, humanize } from "@/lib/format";

import { EvidenceBadge, StatusBadge } from "./badges";
import { ProvenancePanel } from "./provenance";
import { TrajectoryChart } from "./trajectory-chart";

function List({ title, items, testId }: { title: string; items: string[]; testId?: string }) {
  if (!items.length) return null;
  return (
    <section className="text-sm" data-testid={testId}>
      <h3 className="font-medium text-slate-800">{title}</h3>
      <ul className="mt-1 list-disc space-y-1 pl-5 text-slate-700">
        {items.map((x, i) => (
          <li key={i}>{x}</li>
        ))}
      </ul>
    </section>
  );
}

export function Recommendations({ items }: { items: RecommendationView[] }) {
  if (!items.length) return null;
  return (
    <section className="text-sm">
      <h3 className="font-medium text-slate-800">What would let ColdSense answer better</h3>
      <ul className="mt-1 space-y-2">
        {items.map((r, i) => (
          <li key={i} className="rounded border border-slate-200 bg-slate-50 p-2">
            <span className="font-mono text-xs">[{r.priority}]</span> <strong>{r.missing_input}</strong>{" "}
            <span className="text-slate-500">({humanize(r.necessity)})</span>
            <p className="text-slate-700">{r.rationale}</p>
            <p className="text-slate-600">How: {r.acquisition_guidance}</p>
            {r.retry_conditions && <p className="text-slate-600">Retry when: {r.retry_conditions}</p>}
          </li>
        ))}
      </ul>
    </section>
  );
}

function Baselines({ items }: { items: PredictionSuccess["baselines"] }) {
  if (!items.length) return null;
  return (
    <section className="rounded-lg border border-slate-200 bg-white p-4 text-sm" data-testid="baselines">
      <h3 className="font-medium text-slate-800">Baseline comparison (naive reference forecasts from the Phase 4 engine)</h3>
      <table className="mt-2 w-full text-left">
        <thead className="text-xs text-slate-500">
          <tr>
            <th className="py-1">Method</th>
            <th>At horizon</th>
            <th>Crossing</th>
            <th>Held-out MAE (select / test)</th>
          </tr>
        </thead>
        <tbody>
          {items.map((b) => (
            <tr key={b.method} className="border-t border-slate-100">
              <td className="py-1">
                <code>{b.method}</code>
              </td>
              <td>{b.applicable ? formatTemp(b.temp_at_horizon_c) : <span className="text-slate-500">not applicable{b.reason ? `: ${b.reason}` : ""}</span>}</td>
              <td>{b.crossing_status ? <code>{b.crossing_status}</code> : "—"}{b.crossing_minutes != null && ` (+${formatMinutes(b.crossing_minutes)})`}</td>
              <td>
                {formatTemp(b.selection_mae_c)} / {formatTemp(b.test_mae_c)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}

const RELIABILITY_TONE = { HIGH: "good", MEDIUM: "warn", LOW: "bad", UNSUPPORTED: "bad" } as const;

export function ForecastView({ p }: { p: PredictionSuccess }) {
  const r = p.reliability;
  const iv = p.interval;
  return (
    <div className="space-y-4" data-testid="forecast">
      <section className="rounded-lg border border-slate-200 bg-white p-5">
        <div className="flex flex-wrap items-center gap-2">
          <h2 className="text-lg font-semibold text-slate-900">Forecast</h2>
          <EvidenceBadge value={p.evidence_class} label="result" />
          {r && <StatusBadge value={`RELIABILITY ${r.status}`} t={RELIABILITY_TONE[r.status]} />}
        </div>
        <dl className="mt-4 grid grid-cols-1 gap-x-6 gap-y-2 text-sm sm:grid-cols-[max-content_1fr]">
          <dt className="text-slate-500">Question</dt>
          <dd>{p.question === "NO_COOLING" ? "Room air if there is no active cooling from the start" : "Room air if the observed behaviour continues"}</dd>
          <dt className="text-slate-500">Start</dt>
          <dd>
            {formatTimestamp(p.prediction_start)}, last observed {formatTemp(p.initial_temperature.value_c)}{" "}
            <span className="text-slate-500">({p.initial_temperature.provenance})</span>
          </dd>
          <dt className="text-slate-500">Horizon</dt>
          <dd>
            {formatMinutes(p.horizon_min)}
            {p.validated_horizon_min !== null && <span className="text-slate-500"> (validated up to {formatMinutes(p.validated_horizon_min)})</span>}
          </dd>
          <dt className="text-slate-500">Model</dt>
          <dd>
            <code>{p.model.name}</code> {p.model.version && <span className="text-slate-500">v{p.model.version}</span>}
            {p.model.forecast_kind && <span className="text-slate-500"> · {p.model.forecast_kind}</span>}
          </dd>
          {p.threshold && (
            <>
              <dt className="text-slate-500">Threshold</dt>
              <dd>
                {formatTemp(p.threshold.value_c, 1)} {p.threshold.direction.toLowerCase()} ({p.threshold.provenance}, {p.threshold.note})
              </dd>
              <dt className="text-slate-500">Crossing</dt>
              <dd data-testid="crossing">
                <code>{p.crossing.status}</code>
                {p.crossing.time && (
                  <>
                    {" "}
                    at {formatTimestamp(p.crossing.time)} (+{formatMinutes(p.crossing.minutes)})
                  </>
                )}
                {p.crossing_interval && <span className="text-slate-500"> · crossing-time interval: {p.crossing_interval.status}</span>}
              </dd>
            </>
          )}
          {r && (
            <>
              <dt className="text-slate-500">Reliability</dt>
              <dd>
                {r.status} <span className="text-slate-500">— a rule-based classification, not a probability. Scope {r.validation_scope}.</span>
                {r.reason && <span className="block text-slate-600">Below HIGH because: {r.reason}</span>}
              </dd>
            </>
          )}
          {iv && (
            <>
              <dt className="text-slate-500">Interval</dt>
              <dd>
                {iv.lower_c !== null && iv.upper_c !== null ? (
                  <>
                    {formatTemp(iv.lower_c)} … {formatTemp(iv.upper_c)} at +{formatMinutes(iv.horizon_min)} ({Math.round(iv.nominal_coverage * 100)}% nominal, {iv.status})
                  </>
                ) : (
                  <>
                    none: <code>{iv.status}</code>
                  </>
                )}
              </dd>
            </>
          )}
        </dl>
      </section>
      <TrajectoryChart points={p.trajectory} thresholdC={p.threshold?.value_c ?? null} crossingMinutes={p.crossing.minutes} />
      <Baselines items={p.baselines} />
      <List title="Scenario (assumed over the horizon)" items={p.scenario} />
      <List title="Assumptions" items={p.assumptions} />
      <List title="Warnings" items={p.warnings} />
      <List title="Limitations" items={p.limitations} />
      <Recommendations items={p.recommendations} />
      <ProvenancePanel p={p.provenance} />
    </div>
  );
}

export function RefusalPanel({ p }: { p: PredictionRefusal }) {
  return (
    <div className="space-y-4" data-testid="refusal">
      <section role="status" data-state="refusal" className="rounded-lg border border-amber-300 bg-amber-50 p-5">
        <div className="flex flex-wrap items-center gap-2">
          <h2 className="text-lg font-semibold text-amber-950">ColdSense did not make a forecast</h2>
          <StatusBadge value={p.refusal.code} t="warn" />
        </div>
        <p className="mt-2 text-sm text-amber-950">{p.refusal.reason}</p>
        <dl className="mt-3 grid grid-cols-1 gap-x-6 gap-y-2 text-sm sm:grid-cols-[max-content_1fr]">
          {p.refusal.failed_condition && (
            <>
              <dt className="text-amber-800">Condition not met</dt>
              <dd className="text-amber-950">{p.refusal.failed_condition}</dd>
            </>
          )}
          {p.refusal.needed_to_retry && (
            <>
              <dt className="text-amber-800">Needed to retry</dt>
              <dd className="text-amber-950">{p.refusal.needed_to_retry}</dd>
            </>
          )}
          <dt className="text-amber-800">Data verdict</dt>
          <dd className="text-amber-950">{p.data_verdict}</dd>
        </dl>
        <p className="mt-3 text-xs text-amber-800">
          A refusal is a valid result: the evidence does not support an answer, so no trajectory, crossing time or reliability is shown.
        </p>
      </section>
      <List title="Regime evidence" items={p.regime_evidence} />
      <List title="Warnings" items={p.warnings} />
      <List title="Limitations" items={p.limitations} />
      <Recommendations items={p.recommendations} />
      <ProvenancePanel p={p.provenance} />
    </div>
  );
}
