// Typed client for the ColdSense API. Always same-origin (/api/v1/...): the Next.js route handler in
// app/api/[...path] forwards to the FastAPI service, so the session cookie never crosses origins.
import type {
  AnalysisView,
  CaseStatus,
  ErrorBody,
  FindingDraft,
  FindingDraftCreated,
  FindingOutcome,
  FindingsView,
  FindingRecord,
  Health,
  KnowledgeView,
  PredictionRequest,
  PredictionResponse,
  QualityView,
  AnalysisRequest,
  DeclaredEvidence,
  EvidenceView,
  IncidentView,
  InvestigationView,
  NextCheckView,
  ReportView,
  SampleInfo,
  TimelineView,
} from "./types";

export const UPLOAD_MAX_BYTES = 10 * 1024 * 1024; // mirrors coldsense.app.api.limits.UPLOAD_MAX_BYTES
export const UPLOAD_SUFFIXES = [".csv", ".txt"];

export class ApiFailure extends Error {
  constructor(
    readonly status: number,
    readonly body: ErrorBody,
  ) {
    super(body.message);
    this.name = "ApiFailure";
  }
  get code() {
    return this.body.code;
  }
}

/** Codes after which the stored case can no longer be used. */
export const CASE_GONE = new Set(["CASE_NOT_FOUND", "CASE_EXPIRED", "SESSION_REQUIRED", "SESSION_INVALID", "SESSION_EXPIRED", "SESSION_REVOKED"]);

// CSRF (spec §4, amendment G): the API sets an HttpOnly nonce cookie and returns the matching token from GET /csrf.
// The token is kept in memory only and sent as X-CSRF-Token; the browser adds the Origin header itself.
let csrfToken: Promise<string> | null = null;

function getCsrfToken(fresh = false): Promise<string> {
  if (fresh || !csrfToken) {
    csrfToken = call<{ csrf_token: string }>("/csrf").then(
      (r) => r.csrf_token,
      (e) => {
        csrfToken = null;
        throw e;
      },
    );
  }
  return csrfToken;
}

/** A state-changing request: carries the CSRF token; refreshes it once if the server rejects it. */
async function write<T>(path: string, init: RequestInit, retried = false): Promise<T> {
  const token = await getCsrfToken(retried);
  try {
    return await call<T>(path, { ...init, headers: { ...(init.headers as Record<string, string>), "X-CSRF-Token": token } });
  } catch (e) {
    if (!retried && e instanceof ApiFailure && (e.code === "CSRF_TOKEN_INVALID" || e.code === "CSRF_TOKEN_MISSING")) {
      return write<T>(path, init, true);
    }
    throw e;
  }
}

export function resetCsrfForTests() {
  csrfToken = null;
}

async function call<T>(path: string, init: RequestInit = {}): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`/api/v1${path}`, { credentials: "same-origin", cache: "no-store", ...init });
  } catch {
    throw new ApiFailure(0, { code: "NETWORK_ERROR", message: "the service could not be reached", retryable: true });
  }
  if (res.status === 204) return undefined as T;
  let data: unknown = null;
  try {
    data = await res.json();
  } catch {
    // fall through: a non-JSON body is an unexpected response
  }
  if (!res.ok) {
    const err = (data as { error?: ErrorBody } | null)?.error;
    throw new ApiFailure(
      res.status,
      err ?? { code: "UNEXPECTED_RESPONSE", message: `unexpected response (HTTP ${res.status})`, retryable: false },
    );
  }
  if (data === null) {
    throw new ApiFailure(res.status, { code: "UNEXPECTED_RESPONSE", message: "empty response", retryable: false });
  }
  return data as T;
}

export type UploadOptions = { declared?: DeclaredEvidence; unit?: "C" | "F" | "K"; timezone?: string };

const json = (body: unknown): RequestInit => ({
  method: "POST",
  body: JSON.stringify(body),
  headers: { "Content-Type": "application/json" },
});
const id = encodeURIComponent;

export const api = {
  health: () => call<Health>("/health"),
  knowledge: (query: string, k = 5) =>
    call<KnowledgeView>(`/knowledge?${new URLSearchParams({ q: query, k: String(k) })}`),
  samples: () => call<SampleInfo[]>("/samples"),
  createCase(file: File, opts: UploadOptions = {}) {
    const q = new URLSearchParams({ filename: file.name });
    if (opts.declared) q.set("declared_evidence", opts.declared);
    if (opts.unit) q.set("unit", opts.unit);
    if (opts.timezone) q.set("timezone", opts.timezone);
    return write<CaseStatus>(`/cases?${q}`, { method: "POST", body: file, headers: { "Content-Type": "text/csv" } });
  },
  createSampleCase: (sampleId: string) => write<CaseStatus>(`/cases/samples/${id(sampleId)}`, { method: "POST" }),
  endSession: () => write<void>("/session", { method: "DELETE" }),
  getCase: (caseId: string) => call<CaseStatus>(`/cases/${id(caseId)}`),
  deleteCase: (caseId: string) => write<void>(`/cases/${id(caseId)}`, { method: "DELETE" }),
  getQuality: (caseId: string) => call<QualityView>(`/cases/${id(caseId)}/quality`),
  runAnalysis: (caseId: string, body: AnalysisRequest) => write<AnalysisView>(`/cases/${id(caseId)}/analysis`, json(body)),
  getAnalysis: (caseId: string) => call<AnalysisView>(`/cases/${id(caseId)}/analysis`),
  predict: (caseId: string, body: PredictionRequest) => write<PredictionResponse>(`/cases/${id(caseId)}/prediction`, json(body)),
  getPrediction: (caseId: string) => call<PredictionResponse>(`/cases/${id(caseId)}/prediction`),
  getTimeline: (caseId: string) => call<TimelineView>(`/cases/${id(caseId)}/timeline`),
  getEvidence: (caseId: string) => call<EvidenceView>(`/cases/${id(caseId)}/evidence`),
  getNextCheck: (caseId: string) => call<NextCheckView>(`/cases/${id(caseId)}/next-check`),
  getInvestigation: (caseId: string) => call<InvestigationView>(`/cases/${id(caseId)}/investigation`),
  getIncident: (caseId: string) => call<IncidentView>(`/cases/${id(caseId)}/incident`),
  getFindings: (caseId: string) => call<FindingsView>(`/cases/${id(caseId)}/findings`),
  createFindingDraft: (caseId: string, body: { statement: string; event_window: string | null }) =>
    write<FindingDraftCreated>(`/cases/${id(caseId)}/findings/drafts`, json(body)),
  confirmFinding: (caseId: string, draftId: string, body: { token: string; outcome: FindingOutcome }) =>
    write<FindingRecord>(`/cases/${id(caseId)}/findings/drafts/${id(draftId)}/confirm`, json(body)),
  cancelFinding: (caseId: string, draftId: string) =>
    write<FindingDraft>(`/cases/${id(caseId)}/findings/drafts/${id(draftId)}/cancel`, { method: "POST" }),
  createReport: (caseId: string) => write<ReportView>(`/cases/${id(caseId)}/reports`, { method: "POST" }),
  getLatestReport: (caseId: string) => call<ReportView>(`/cases/${id(caseId)}/reports/latest`),
};

export type Api = typeof api;

/** Checks done before sending, so an obviously invalid file never leaves the browser. */
export function checkFile(file: File): string | null {
  const name = file.name.toLowerCase();
  if (!UPLOAD_SUFFIXES.some((s) => name.endsWith(s))) return `Only ${UPLOAD_SUFFIXES.join(", ")} files are accepted.`;
  if (file.size === 0) return "The file is empty.";
  if (file.size > UPLOAD_MAX_BYTES) return `The file is larger than ${UPLOAD_MAX_BYTES / (1024 * 1024)} MiB.`;
  return null;
}
