// Aliases over the types generated from the backend's OpenAPI document (schema.d.ts: do not edit by hand;
// regenerate with `npm run gen:types` after `python -m coldsense.app.api.openapi_export web/openapi.json`).
import type { components } from "./schema";

type S = components["schemas"];

export type CaseStatus = S["CaseStatus"];
export type QualityView = S["QualityView"];
export type AnalysisView = S["AnalysisView"];
export type PredictionRequest = S["PredictionRequest-Input"];
export type PredictionSuccess = S["PredictionSuccess"];
export type PredictionRefusal = S["PredictionRefusal"];
export type PredictionResponse = PredictionSuccess | PredictionRefusal;
export type Health = S["Health"];
export type KnowledgeView = S["KnowledgeView"];
export type ResultProvenance = S["ResultProvenance"];
export type RecommendationView = S["RecommendationView"];
export type Issue = S["Issue"];
export type EvidenceClass = S["EvidenceClass"];
export type StepState = S["StepState"];
export type Verdict = S["Verdict"];
export type TrajectoryPoint = S["TrajectoryPointView"];
export type DeclaredEvidence = "REAL" | "SIMULATED";
export type SampleInfo = S["SampleInfo"];
export type AnalysisRequest = S["AnalysisRequest"];
export type BaselineView = S["BaselineView"];
export type TimelineView = S["TimelineView"];
export type TimelineRun = S["TimelineRun"];
export type TimelineGap = S["TimelineGap"];
export type PhaseEvent = S["Event"];
export type EvidenceView = S["EvidenceView"];
export type NextCheckView = S["NextCheckView"];
export type InvestigationView = S["InvestigationView"];
export type IncidentView = S["IncidentView"];
export type FindingDraft = S["FindingDraftView"];
export type FindingDraftCreated = S["FindingDraftCreated"];
export type FindingRecord = S["FindingRecordView"];
export type FindingCounts = S["FindingCounts"];
export type FindingsView = S["FindingsView"];
export type FindingOutcome = "UNKNOWN" | "RESOLVED" | "NOT_RESOLVED";
export type ReportView = S["ReportView"];
export type Finding = S["Finding"];
export type NextCheck = S["NextCheck"];
export type Hypothesis = S["Hypothesis"];
export type ObservedFact = S["ObservedFact"];
export type DiagnosisStatus = S["DiagnosisStatus"];

/** Codes the backend can return, plus the proxy's own (the backend is not reachable). */
export type ErrorCode = S["ErrorCode"] | "BACKEND_UNAVAILABLE" | "NETWORK_ERROR" | "UNEXPECTED_RESPONSE";
export type ErrorBody = { code: ErrorCode; message: string; retryable: boolean };

export function isRefusal(p: PredictionResponse): p is PredictionRefusal {
  return p.outcome === "REFUSED";
}
