"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

import { ErrorState, LoadingState } from "@/components/states";
import { ApiFailure, checkFile, UPLOAD_MAX_BYTES } from "@/lib/api/client";
import type { CaseStatus, DeclaredEvidence, SampleInfo } from "@/lib/api/types";
import { useCase } from "@/lib/case-context";

const DECLARATIONS: { value: "" | DeclaredEvidence; label: string }[] = [
  { value: "", label: "No declaration" },
  { value: "REAL", label: "REAL — I say it was measured on a physical cold room" },
  { value: "SIMULATED", label: "SIMULATED — I say it was produced by a simulator" },
];

const asFailure = (e: unknown) =>
  e instanceof ApiFailure ? e : new ApiFailure(0, { code: "UNEXPECTED_RESPONSE", message: String(e), retryable: false });

export default function UploadPage() {
  const { api, setStatus } = useCase();
  const router = useRouter();
  const [file, setFile] = useState<File | null>(null);
  const [declared, setDeclared] = useState<"" | DeclaredEvidence>("");
  const [unit, setUnit] = useState<"" | "C" | "F" | "K">("");
  const [timezone, setTimezone] = useState("");
  const [busy, setBusy] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);
  const [error, setError] = useState<ApiFailure | null>(null);
  const [samples, setSamples] = useState<SampleInfo[]>([]);

  useEffect(() => {
    api.samples().then(setSamples, () => setSamples([]));
  }, [api]);

  async function start(make: () => Promise<CaseStatus>) {
    setError(null);
    setBusy(true);
    try {
      setStatus(await make());
      router.push("/quality");
    } catch (err) {
      setError(asFailure(err));
    } finally {
      setBusy(false);
    }
  }

  function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!file) return setLocalError("Choose a file first.");
    const problem = checkFile(file);
    setLocalError(problem);
    if (problem) return;
    void start(() => api.createCase(file, { declared: declared || undefined, unit: unit || undefined, timezone: timezone.trim() || undefined }));
  }

  return (
    <div className="space-y-4">
      <h1 className="text-xl font-semibold">Upload Data</h1>
      <p className="text-sm text-slate-600">
        A CSV log with a timestamp and a temperature column; ambient temperature and compressor state are optional. Accepted: .csv or .txt, at most{" "}
        {UPLOAD_MAX_BYTES / (1024 * 1024)} MiB and 200 000 rows. The file is kept in memory for this case only, is never written to disk by ColdSense, and is
        deleted when the case expires (2 h) or when you delete it.
      </p>
      <form onSubmit={submit} className="space-y-4 rounded-lg border border-slate-200 bg-white p-5" aria-busy={busy}>
        <div>
          <label htmlFor="file" className="block text-sm font-medium">
            Log file
          </label>
          <input
            id="file"
            name="file"
            type="file"
            accept=".csv,.txt,text/csv,text/plain"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            className="mt-1 block w-full text-sm file:mr-3 file:rounded-md file:border-0 file:bg-slate-100 file:px-3 file:py-1.5"
          />
        </div>
        <fieldset>
          <legend className="text-sm font-medium">Where does this data come from? (optional)</legend>
          <p className="text-xs text-slate-600">
            Uploads are always analysed as <strong>UNVALIDATED</strong>. What you declare is stored and shown as &ldquo;declared by uploader, not
            verified&rdquo;; it does not change any result.
          </p>
          <div className="mt-1 space-y-1">
            {DECLARATIONS.map((o) => (
              <label key={o.value || "none"} className="flex items-center gap-2 text-sm">
                <input type="radio" name="declared" value={o.value} checked={declared === o.value} onChange={() => setDeclared(o.value)} />
                {o.label}
              </label>
            ))}
          </div>
        </fieldset>
        <details className="text-sm">
          <summary className="cursor-pointer text-slate-700">Optional: unit and time zone (when the file does not say)</summary>
          <div className="mt-2 grid gap-3 sm:grid-cols-2">
            <label className="block">
              Temperature unit
              <select value={unit} onChange={(e) => setUnit(e.target.value as typeof unit)} className="mt-1 block w-full rounded border border-slate-300 px-2 py-1">
                <option value="">detect from the file</option>
                <option value="C">°C</option>
                <option value="F">°F</option>
                <option value="K">K</option>
              </select>
            </label>
            <label className="block">
              Time zone of naive timestamps (IANA, e.g. Africa/Algiers)
              <input value={timezone} onChange={(e) => setTimezone(e.target.value)} maxLength={64} className="mt-1 block w-full rounded border border-slate-300 px-2 py-1" />
            </label>
          </div>
        </details>
        {localError && (
          <p role="alert" className="text-sm text-rose-800">
            {localError}
          </p>
        )}
        <button type="submit" disabled={busy} className="rounded-md bg-sky-700 px-4 py-2 text-sm font-medium text-white hover:bg-sky-800 disabled:opacity-50">
          Upload and check data quality
        </button>
      </form>
      {samples.length > 0 && (
        <section className="rounded-lg border border-sky-200 bg-sky-50 p-5 text-sm" data-testid="samples">
          <h2 className="font-semibold text-sky-950">Or try a bundled ColdSense sample</h2>
          <p className="mt-1 text-sky-900">These files ship with ColdSense and are labelled SIMULATED. They are never evidence about a real room.</p>
          <ul className="mt-3 space-y-2">
            {samples.map((s) => (
              <li key={s.sample_id} className="flex flex-wrap items-center gap-3">
                <button
                  type="button"
                  disabled={busy}
                  onClick={() => void start(() => api.createSampleCase(s.sample_id))}
                  className="rounded-md border border-sky-400 bg-white px-3 py-1 font-medium text-sky-900 hover:bg-sky-100 disabled:opacity-50"
                >
                  Use {s.filename}
                </button>
                <span className="text-sky-900">{s.description}</span>
              </li>
            ))}
          </ul>
        </section>
      )}
      {busy && <LoadingState label="Uploading and checking data quality…" />}
      {error && <ErrorState error={error} />}
    </div>
  );
}
