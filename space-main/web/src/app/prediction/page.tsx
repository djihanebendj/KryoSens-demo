"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { ForecastView, RefusalPanel } from "@/components/prediction-views";
import { EmptyState, ErrorState, LoadingState } from "@/components/states";
import { ApiFailure } from "@/lib/api/client";
import { isRefusal, type PredictionRequest } from "@/lib/api/types";
import { useCase } from "@/lib/case-context";

function toNumber(s: string): number | null {
  if (s.trim() === "") return null;
  const n = Number(s);
  return Number.isFinite(n) ? n : NaN;
}

export default function PredictionPage() {
  const { api, caseId, status, phase, error: caseError, prediction, setPrediction, refresh } = useCase();
  const [horizon, setHorizon] = useState("120");
  const [threshold, setThreshold] = useState("");
  const [direction, setDirection] = useState<PredictionRequest["direction"]>("UPWARD");
  const [question, setQuestion] = useState<PredictionRequest["question"]>("NO_COOLING");
  const [passiveSince, setPassiveSince] = useState("");
  const [reliability, setReliability] = useState(true);
  const [busy, setBusy] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);
  const [error, setError] = useState<ApiFailure | null>(null);

  // Show the case's latest prediction when arriving on the page.
  useEffect(() => {
    if (!caseId || prediction || !status?.last_prediction_outcome) return;
    api.getPrediction(caseId).then(setPrediction, () => undefined);
  }, [api, caseId, status?.last_prediction_outcome, prediction, setPrediction]);

  if (phase === "loading") return <LoadingState label="Loading the case…" />;
  if (phase === "error" && caseError) return <ErrorState error={caseError} />;
  if (!caseId || !status) {
    return (
      <div className="space-y-4">
        <h1 className="text-xl font-semibold">Prediction</h1>
        <EmptyState title="No data uploaded yet" action={{ href: "/upload", label: "Upload Data" }}>
          A prediction needs a log whose data quality allows analysis.
        </EmptyState>
      </div>
    );
  }

  const blocked = !status.quality.suitable_for_analysis;

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    const h = toNumber(horizon);
    const t = toNumber(threshold);
    if (h === null || Number.isNaN(h) || h <= 0) return setFormError("Horizon must be a positive number of minutes.");
    if (Number.isNaN(t)) return setFormError("Threshold must be a number in °C, or empty.");
    setFormError(null);
    setBusy(true);
    try {
      const body: PredictionRequest = {
        horizon_min: h,
        threshold_c: t,
        direction,
        question,
        passive_since: passiveSince || null,
        assess_reliability: reliability,
      };
      setPrediction(await api.predict(caseId!, body));
      void refresh();
    } catch (err) {
      setError(err instanceof ApiFailure ? err : new ApiFailure(0, { code: "UNEXPECTED_RESPONSE", message: String(err), retryable: false }));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-4">
      <h1 className="text-xl font-semibold">Prediction</h1>
      {blocked ? (
        <section role="status" data-state="blocked" className="rounded-lg border border-rose-300 bg-rose-50 p-5 text-sm text-rose-900">
          Prediction is blocked: the data-quality check refused this log ({status.quality.verdict}). See Data Quality for the reasons.
        </section>
      ) : (
        <form onSubmit={submit} className="grid gap-4 rounded-lg border border-slate-200 bg-white p-5 sm:grid-cols-2" aria-busy={busy}>
          <label className="block text-sm">
            Horizon (minutes)
            <input id="horizon" inputMode="decimal" value={horizon} onChange={(e) => setHorizon(e.target.value)} className="mt-1 block w-full rounded border border-slate-300 px-2 py-1" />
          </label>
          <label className="block text-sm">
            Threshold (°C, optional)
            <input id="threshold" inputMode="decimal" value={threshold} onChange={(e) => setThreshold(e.target.value)} className="mt-1 block w-full rounded border border-slate-300 px-2 py-1" />
          </label>
          <label className="block text-sm">
            Crossing direction
            <select value={direction} onChange={(e) => setDirection(e.target.value as typeof direction)} className="mt-1 block w-full rounded border border-slate-300 px-2 py-1">
              <option value="UPWARD">rising to or above the threshold</option>
              <option value="DOWNWARD">falling to or below the threshold</option>
            </select>
          </label>
          <label className="block text-sm">
            Question
            <select value={question} onChange={(e) => setQuestion(e.target.value as typeof question)} className="mt-1 block w-full rounded border border-slate-300 px-2 py-1">
              <option value="NO_COOLING">no active cooling from the last reading</option>
              <option value="CONTINUATION">observed behaviour continues</option>
            </select>
          </label>
          <label className="block text-sm">
            Cooling stopped at (optional, log time)
            <input type="datetime-local" value={passiveSince} onChange={(e) => setPassiveSince(e.target.value)} className="mt-1 block w-full rounded border border-slate-300 px-2 py-1" />
          </label>
          <label className="flex items-center gap-2 self-end text-sm">
            <input type="checkbox" checked={reliability} onChange={(e) => setReliability(e.target.checked)} />
            Assess reliability (Phase 5)
          </label>
          {formError && (
            <p role="alert" className="text-sm text-rose-800 sm:col-span-2">
              {formError}
            </p>
          )}
          <div className="sm:col-span-2">
            <button type="submit" disabled={busy} className="rounded-md bg-sky-700 px-4 py-2 text-sm font-medium text-white hover:bg-sky-800 disabled:opacity-50">
              Run prediction
            </button>
          </div>
        </form>
      )}
      {busy && <LoadingState label="Running the thermal model and reliability checks… this can take a few seconds." />}
      {error && <ErrorState error={error} />}
      {!busy && prediction && (isRefusal(prediction) ? <RefusalPanel p={prediction} /> : <ForecastView p={prediction} />)}
      {!busy && prediction && (
        <p className="text-sm text-slate-700" data-testid="next-step">
          Next: compare with the recorded evidence on <Link href="/evidence" className="text-sky-800 underline">Evidence</Link> and the ranked checks on{" "}
          <Link href="/next-check" className="text-sky-800 underline">Next Best Check</Link> (both need the analysis on Temperature Timeline).
        </p>
      )}
    </div>
  );
}
