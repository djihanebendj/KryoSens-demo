// Which sections and journey steps are usable for the current case, and why not. Derived only from the backend's
// case status (steps READY / REFUSED / NOT_RUN / BLOCKED); nothing is computed here.
import type { CaseStatus } from "./api/types";

export type Gate = { enabled: true } | { enabled: false; reason: string };

const ok: Gate = { enabled: true };
const no = (reason: string): Gate => ({ enabled: false, reason });

export function hasAnalysis(s: CaseStatus | null): boolean {
  return !!s && (s.steps.analysis === "READY" || s.steps.analysis === "REFUSED");
}

/** Prerequisites per section href (spec §8: disable steps whose prerequisites are missing). */
export function gateFor(href: string, s: CaseStatus | null): Gate {
  switch (href) {
    case "/":
    case "/upload":
      return ok;
    case "/copilot":
      return ok; // states its own FEATURE_DISABLED reason from the server gate
    case "/reports":
    case "/quality":
    case "/incident":
      return s ? ok : no("upload data first");
    case "/timeline":
    case "/prediction":
      if (!s) return no("upload data first");
      return s.quality.suitable_for_analysis ? ok : no("the data-quality check refused this log");
    case "/evidence":
    case "/next-check":
    case "/investigation":
      if (!s) return no("upload data first");
      if (!s.quality.suitable_for_analysis) return no("the data-quality check refused this log");
      return hasAnalysis(s) ? ok : no("run the analysis on Temperature Timeline first");
    default:
      return ok;
  }
}

export type StepState = "done" | "refused" | "available" | "blocked" | "not-available";

export type JourneyStep = { label: string; href: string | null; state: StepState; note?: string };

/** The main journey of spec §8, with the state of each step for this case. */
export function journey(s: CaseStatus | null): JourneyStep[] {
  const step = (label: string, href: string, done: boolean, refused = false): JourneyStep => {
    const g = gateFor(href, s);
    return { label, href, state: !g.enabled ? "blocked" : refused ? "refused" : done ? "done" : "available", note: g.enabled ? undefined : g.reason };
  };
  return [
    step("Upload data", "/upload", !!s),
    step("Data quality", "/quality", !!s && s.steps.quality === "READY", !!s && s.steps.quality === "REFUSED"),
    step("Analysis", "/timeline", hasAnalysis(s), !!s && s.steps.analysis === "REFUSED"),
    step("Prediction", "/prediction", !!s && s.steps.prediction === "READY", !!s && s.steps.prediction === "REFUSED"),
    step("Evidence", "/evidence", false),
    step("Next best check", "/next-check", false),
    { label: "Copilot", href: "/copilot", state: "not-available", note: "FEATURE_DISABLED" },
    step("Technician finding", "/incident", false),
    step("Verification", "/incident", false),
    step("Report", "/reports", false),
  ];
}
