"use client";

// The Copilot page reads only the server's health/gate state. It never calls the copilot endpoint while the gate is
// closed, constructs no provider and makes no LLM or tool call. The reason shown is the server's own reason.
import { useEffect, useState } from "react";

import { StatusBadge } from "@/components/badges";
import { ErrorState, LoadingState } from "@/components/states";
import { ApiFailure } from "@/lib/api/client";
import type { Health } from "@/lib/api/types";
import { useCase } from "@/lib/case-context";

export function CopilotStatus() {
  const { api } = useCase();
  const [health, setHealth] = useState<Health | null>(null);
  const [error, setError] = useState<ApiFailure | null>(null);

  useEffect(() => {
    let live = true;
    api.health().then(
      (h) => { if (live) setHealth(h); },
      (e: unknown) => {
        if (!live) return;
        setError(e instanceof ApiFailure ? e : new ApiFailure(0, {
          code: "UNEXPECTED_RESPONSE", message: "the request failed", retryable: false,
        }));
      },
    );
    return () => { live = false; };
  }, [api]);

  if (error) return <ErrorState error={error} />;
  if (!health) return <LoadingState label="Checking the server-side Copilot gate…" />;

  const enabled = health.copilot === "ENABLED";
  return (
    <div className="space-y-4">
      <header className="space-y-1">
        <h1 className="text-2xl font-semibold tracking-tight">Technician Copilot</h1>
        <p className="text-sm text-slate-600">
          The server decides whether this capability exists. A hidden button is not the lock: a direct API request is
          refused by the same gate.
        </p>
      </header>

      <section className="rounded-lg border border-slate-200 bg-white p-5" data-testid="copilot-status">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-sm font-medium">Server state</span>
          <StatusBadge value={health.copilot} t={enabled ? "good" : "bad"} />
          <span className="text-xs text-slate-500">ColdSense {health.coldsense_version}</span>
        </div>
        <p data-testid="copilot-reason" className="mt-3 rounded-md bg-slate-50 p-3 text-sm text-slate-800">
          {health.copilot_reason}
        </p>
      </section>

      <section className="space-y-2 rounded-lg border border-amber-300 bg-amber-50 p-5 text-sm text-amber-950">
        <h2 className="font-semibold">Why the Copilot is disabled</h2>
        <p>
          The gate needs explicit deployment configuration, an LLM provider for the Phase 8 adapter, and a verified
          Phase 8 acceptance record for the Rule 7 hardening. No acceptance record exists in this repository, and this
          Phase 9 slice deliberately does not wire a conversation provider, so the state stays{" "}
          <code className="rounded bg-amber-100 px-1">FEATURE_DISABLED</code>.
        </p>
        <p>
          While disabled, no language model is called and no copilot tool runs. This page makes only GET requests
          (<code className="rounded bg-amber-100 px-1">/api/v1/health</code> and, for Guided Start, the case&apos;s
          next-check, investigation and evidence views); it never sends a question to the copilot endpoint.
        </p>
      </section>

      <section className="space-y-2 text-sm text-slate-700">
        <h2 className="font-semibold">What the disabled state means</h2>
        <ul className="list-disc space-y-1 pl-5">
          <li>There is no conversational answer, diagnosis or recommendation on this page.</li>
          <li>No LLM-generated prose is displayed as a finding or a measurement.</li>
          <li>Technician findings are entered by you on Incident and confirmed by you, never by the copilot.</li>
          <li>The evidence, next-check and investigation pages remain the model-inference surfaces, with their labels.</li>
        </ul>
      </section>
    </div>
  );
}
