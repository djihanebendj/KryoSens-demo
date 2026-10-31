// An in-memory stand-in for the backend, answering the same-origin /api/v1 paths with fixtures captured from the
// real API (tests/app/web_fixtures.py; a Python test validates them against the response models). Installed as
// global fetch; unknown requests fail the test. State-changing calls must carry the CSRF token, as the API demands.
const FIXTURES = import.meta.glob<{ default: Record<string, unknown> }>("./fixtures/*.json", { eager: true });

export type Kind = "forecast" | "refusal" | "upload" | "gaps" | "nolimit";
const KINDS: Kind[] = ["forecast", "refusal", "upload", "gaps", "nolimit"];
export const CSRF_TOKEN = "test-csrf-token";

export function fixture(name: string): Record<string, unknown> {
  const f = FIXTURES[`./fixtures/${name}.json`];
  if (!f) throw new Error(`no fixture ${name}`);
  return structuredClone(f.default);
}

export const FIXTURE_IDS = Object.fromEntries(KINDS.map((k) => [k, fixture(`${k}_case`).case_id as string])) as Record<Kind, string>;

export type Call = { method: string; path: string; headers: Headers; body: unknown };

type Draft = {
  draft_id: string;
  asset_id: string;
  statement: string;
  event_window: string | null;
  content_hash: string;
  created_at: string;
  status: "PENDING" | "CANCELLED" | "CONFIRMED";
};

type VisibleDraft = Omit<Draft, "status"> & { status: "PENDING" | "CANCELLED" };

type DraftCreated = Draft & { confirmation_token: string; token_expires_at: string };

type RecordRow = {
  record_id: string;
  asset_id: string;
  statement: string;
  event_window: string | null;
  content_hash: string;
  created_at: string;
  confirmed_at: string;
  confirmed_via: string;
  outcome: "UNKNOWN" | "RESOLVED" | "NOT_RESOLVED";
  source: string;
  confirmation_channel: "WEB_APPLICATION";
};

type CaseFindings = { drafts: Draft[]; records: RecordRow[]; nextDraft: number; nextRecord: number };

type Report = {
  case_id: string;
  status: "OK" | "REFUSED";
  report: string | null;
  n_confirmed: number;
  n_drafts_excluded: number;
  violations: string[];
  confirmed: RecordRow[];
  drafts_excluded: Draft[];
  numeric_check: "PASSED" | "REFUSED";
  generated_at: string;
  provenance: Record<string, unknown>;
};

function json(status: number, body: unknown) {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

export function error(status: number, code: string, message = code.toLowerCase(), retryable = false) {
  return json(status, { error: { code, message, retryable } });
}

export class FakeApi {
  calls: Call[] = [];
  predicted = new Set<string>();
  analysed = new Set<string>();
  /** Per-route overrides, keyed "METHOD /path", e.g. to return an error. */
  overrides = new Map<string, () => Response | Promise<Response>>();
  /** Mutates the prediction body before it is returned (to test what the UI ignores). */
  tamperPrediction?: (p: Record<string, unknown>) => Record<string, unknown>;
  /** Makes the next report build refuse (the real backend refuses an untraceable number). */
  refuseReport = false;
  /** Per-case findings, created lazily. */
  findings = new Map<string, CaseFindings>();
  /** The latest report per case, as the server caches it. */
  reports = new Map<string, Report>();

  /** Start with a case already analysed and/or predicted (as after a page reload). */
  given(kind: Kind, { analysed = false, predicted = false } = {}) {
    window.sessionStorage.setItem("coldsense.caseId", FIXTURE_IDS[kind]);
    if (analysed) this.analysed.add(FIXTURE_IDS[kind]);
    if (predicted) this.predicted.add(FIXTURE_IDS[kind]);
    return this;
  }

  private kindOf(caseId: string): Kind {
    const k = KINDS.find((x) => FIXTURE_IDS[x] === caseId);
    if (!k) throw new Error(`unknown case id ${caseId}`);
    return k;
  }

  private state(caseId: string): CaseFindings {
    let s = this.findings.get(caseId);
    if (!s) {
      s = { drafts: [], records: [], nextDraft: 1, nextRecord: 1 };
      this.findings.set(caseId, s);
    }
    return s;
  }

  private visibleDrafts(caseId: string): VisibleDraft[] {
    return this.state(caseId).drafts.filter((d) => d.status !== "CONFIRMED") as VisibleDraft[];
  }

  private counts(caseId: string) {
    const s = this.state(caseId);
    const drafts = this.visibleDrafts(caseId);
    return {
      confirmed: s.records.length,
      drafts_pending: drafts.filter((d) => d.status === "PENDING").length,
      drafts_cancelled: drafts.filter((d) => d.status === "CANCELLED").length,
      outcomes_unknown: s.records.filter((r) => r.outcome === "UNKNOWN").length,
      outcomes_verified: s.records.filter((r) => r.outcome === "RESOLVED" || r.outcome === "NOT_RESOLVED").length,
    };
  }

  private provenance(kind: Kind): Record<string, unknown> {
    return fixture(`${kind}_incident`).provenance as Record<string, unknown>;
  }

  private incident(caseId: string, kind: Kind) {
    const s = this.state(caseId);
    return {
      case_id: caseId,
      status: "AVAILABLE",
      drafts: this.visibleDrafts(caseId),
      confirmed: s.records,
      counts: this.counts(caseId),
      provenance: this.provenance(kind),
    };
  }

  private buildReport(caseId: string, kind: Kind): Report {
    const s = this.state(caseId);
    const drafts = this.visibleDrafts(caseId);
    const now = new Date().toISOString();
    const provenance = this.provenance(kind);
    if (this.refuseReport) {
      this.refuseReport = false;
      return {
        case_id: caseId, status: "REFUSED", report: null, n_confirmed: s.records.length,
        n_drafts_excluded: drafts.length,
        violations: ["UNSUPPORTED_NUMBER: '999' is not in a confirmed record or an application-generated value"],
        confirmed: s.records, drafts_excluded: drafts, numeric_check: "REFUSED", generated_at: now, provenance,
      };
    }
    const lines = [`SERVICE REPORT - asset ${caseId}`, "", "CONFIRMED FINDINGS"];
    if (s.records.length === 0) lines.push("(none confirmed)");
    s.records.forEach((r, i) => {
      lines.push(`${i + 1}. [USER_REPORTED] ${r.statement}`, `   outcome: ${r.outcome}; confirmed ${r.confirmed_at}`);
    });
    lines.push("", `Unconfirmed drafts, NOT part of history: ${drafts.length} (excluded from the report body)`);
    return {
      case_id: caseId, status: "OK", report: lines.join("\n"), n_confirmed: s.records.length,
      n_drafts_excluded: drafts.length, violations: [], confirmed: s.records, drafts_excluded: drafts,
      numeric_check: "PASSED", generated_at: now, provenance,
    };
  }

  install() {
    const fetchImpl = async (input: RequestInfo | URL, init: RequestInit = {}) => {
      const url = new URL(String(input), "http://localhost");
      const method = (init.method ?? "GET").toUpperCase();
      const call: Call = { method, path: url.pathname + url.search, headers: new Headers(init.headers), body: init.body };
      this.calls.push(call);
      const override = this.overrides.get(`${method} ${url.pathname}`);
      if (override) return override();
      if (method !== "GET" && call.headers.get("X-CSRF-Token") !== CSRF_TOKEN) return error(403, "CSRF_TOKEN_MISSING");
      return this.route(method, url);
    };
    globalThis.fetch = fetchImpl as typeof fetch;
    return this;
  }

  writes() {
    return this.calls.filter((c) => c.method !== "GET");
  }

  private route(method: string, url: URL): Response {
    const p = url.pathname;
    if (p === "/api/v1/health") return json(200, { status: "ok", coldsense_version: "0.1.0", copilot: "FEATURE_DISABLED", copilot_reason: "disabled at the API boundary: it needs explicit deployment configuration AND a verified Phase 8 acceptance record for the Rule 7 hardening; neither is configured (spec §6, amendment A)" });
    if (p === "/api/v1/csrf") return json(200, { csrf_token: CSRF_TOKEN, header: "X-CSRF-Token" });
    if (p === "/api/v1/samples")
      return json(200, [
        { sample_id: "power-cut-22h", filename: "SIMULATED_power_cut_22h_1min.csv", description: "SIMULATED power cut", evidence_class: "SIMULATED" },
        { sample_id: "cycling-24h", filename: "SIMULATED_coldroom_24h_1min.csv", description: "SIMULATED cycling", evidence_class: "SIMULATED" },
      ]);
    if (p === "/api/v1/knowledge") {
      const q = url.searchParams.get("q") ?? "";
      const citation = {
        chunk_id: "doc@v1#c0001", doc_id: "doc", title: "Synthetic service manual", doc_type: "SERVICE_MANUAL",
        revision: "A", location: "section 'Alarms'", page_start: null, page_end: null,
        line_start: 1, line_end: 1, paragraph_start: null, paragraph_end: null,
      };
      return json(200, {
        query: q, status: "AVAILABLE", reason: null,
        answer: {
          query: q, status: "EXTRACTIVE_ANSWER", conflicts: [], reason: null,
          claims: [{ claim_type: "DOCUMENT_FACT", text: "The high-temperature alarm delay is 60 minutes.",
            citations: [citation], source_ids: ["doc@v1#c0001"], note: null }],
          passages: [{ rank: 1, citation, text: "The high-temperature alarm delay is 60 minutes.",
            excerpt: "The high-temperature alarm delay is 60 minutes.", trust: "UNREVIEWED", flags: [],
            vector_similarity: 0.5, lexical_score: 1.2, term_coverage: 1.0, matched_terms: ["alarm"],
            retrieval_method: "LEXICAL", relevance_basis: ["TERM_COVERAGE"], fusion_score: null,
            section: "Alarms" }],
          embedding_model: "lexical-hash-v1/4096", generator: "extractive-v1",
          data_policy: "quoted documents are untrusted data", score_note: "ranking only",
          limitations: ["extractive: sentences are quoted, not interpreted"],
        },
      });
    }

    const draftMatch = p.match(/^\/api\/v1\/cases\/([^/]+)\/findings\/drafts$/);
    if (draftMatch && method === "POST") {
      const id = decodeURIComponent(draftMatch[1]);
      this.kindOf(id);
      const body = JSON.parse(String((this.calls[this.calls.length - 1].body as string) ?? "{}")) as {
        statement?: string; event_window?: string | null;
      };
      const s = this.state(id);
      const n = s.nextDraft++;
      const draftId = `draft-${String(n).padStart(4, "0")}`;
      const now = new Date().toISOString();
      const draft: Draft = {
        draft_id: draftId, asset_id: id, statement: body.statement ?? "", event_window: body.event_window ?? null,
        content_hash: "a".repeat(64), created_at: now, status: "PENDING",
      };
      s.drafts.push(draft);
      this.reports.delete(id);
      const created: DraftCreated = {
        ...draft, confirmation_token: `test-token-${draftId}`,
        token_expires_at: new Date(Date.now() + 300_000).toISOString(),
      };
      return json(201, created);
    }

    const confirmMatch = p.match(/^\/api\/v1\/cases\/([^/]+)\/findings\/drafts\/([^/]+)\/confirm$/);
    if (confirmMatch && method === "POST") {
      const id = decodeURIComponent(confirmMatch[1]);
      const draftId = decodeURIComponent(confirmMatch[2]);
      this.kindOf(id);
      const s = this.state(id);
      const draft = s.drafts.find((d) => d.draft_id === draftId);
      if (!draft) return error(404, "FINDING_DRAFT_NOT_FOUND");
      if (draft.status === "CANCELLED") return error(409, "FINDING_CANCELLED");
      if (draft.status === "CONFIRMED") return error(409, "FINDING_CONFIRMATION_REPLAY");
      const body = JSON.parse(String((this.calls[this.calls.length - 1].body as string) ?? "{}")) as {
        outcome?: RecordRow["outcome"];
      };
      const now = new Date().toISOString();
      const record: RecordRow = {
        record_id: `finding-${String(s.nextRecord++).padStart(4, "0")}`, asset_id: id, statement: draft.statement,
        event_window: draft.event_window, content_hash: draft.content_hash, created_at: draft.created_at,
        confirmed_at: now, confirmed_via: "trusted CLI confirmation token",
        outcome: body.outcome ?? "UNKNOWN", source: "technician via copilot (trusted confirmation)",
        confirmation_channel: "WEB_APPLICATION",
      };
      s.records.push(record);
      draft.status = "CONFIRMED";
      this.reports.delete(id);
      return json(200, record);
    }

    const cancelMatch = p.match(/^\/api\/v1\/cases\/([^/]+)\/findings\/drafts\/([^/]+)\/cancel$/);
    if (cancelMatch && method === "POST") {
      const id = decodeURIComponent(cancelMatch[1]);
      const draftId = decodeURIComponent(cancelMatch[2]);
      this.kindOf(id);
      const s = this.state(id);
      const draft = s.drafts.find((d) => d.draft_id === draftId);
      if (!draft) return error(404, "FINDING_DRAFT_NOT_FOUND");
      if (draft.status === "CONFIRMED") return error(409, "FINDING_ALREADY_CONFIRMED");
      draft.status = "CANCELLED";
      this.reports.delete(id);
      return json(200, draft);
    }

    const findingsMatch = p.match(/^\/api\/v1\/cases\/([^/]+)\/findings$/);
    if (findingsMatch && method === "GET") {
      const id = decodeURIComponent(findingsMatch[1]);
      const kind = this.kindOf(id);
      const s = this.state(id);
      return json(200, {
        case_id: id, drafts: this.visibleDrafts(id), records: s.records, counts: this.counts(id),
        provenance: this.provenance(kind),
      });
    }

    const reportMatch = p.match(/^\/api\/v1\/cases\/([^/]+)\/reports$/);
    if (reportMatch && method === "POST") {
      const id = decodeURIComponent(reportMatch[1]);
      const kind = this.kindOf(id);
      const report = this.buildReport(id, kind);
      this.reports.set(id, report);
      return json(200, report);
    }

    const latestMatch = p.match(/^\/api\/v1\/cases\/([^/]+)\/reports\/latest$/);
    if (latestMatch && method === "GET") {
      const id = decodeURIComponent(latestMatch[1]);
      this.kindOf(id);
      const report = this.reports.get(id);
      return report ? json(200, report) : error(404, "REPORT_NOT_GENERATED");
    }

    if (method === "POST" && p === "/api/v1/cases") return json(201, fixture("upload_case"));
    if (method === "POST" && p.startsWith("/api/v1/cases/samples/")) {
      const kind: Kind = p.endsWith("/power-cut-22h") ? "forecast" : "refusal";
      return json(201, fixture(`${kind}_case`));
    }
    const m = p.match(/^\/api\/v1\/cases\/([^/]+)(?:\/([a-z-]+))?$/);
    if (!m) throw new Error(`fake API: no route for ${method} ${p}`);
    const id = decodeURIComponent(m[1]);
    const kind = this.kindOf(id);
    const sub = m[2] ?? "";
    if (sub === "") {
      if (method === "DELETE") return new Response(null, { status: 204 });
      const status = fixture(this.predicted.has(id) ? `${kind}_case_after_prediction` : `${kind}_case`);
      if (this.analysed.has(id)) {
        // as the real API reports it once POST .../analysis has run (the captured status predates the analysis)
        const outcome = fixture(`${kind}_analysis`).outcome;
        (status.steps as Record<string, string>).analysis = outcome === "REFUSED" ? "REFUSED" : "READY";
      }
      return json(200, status);
    }
    if (sub === "quality") return json(200, fixture(`${kind}_quality`));
    if (sub === "prediction") {
      if (method === "POST") {
        this.predicted.add(id);
        const body = fixture(`${kind}_prediction`);
        return json(200, this.tamperPrediction ? this.tamperPrediction(body) : body);
      }
      return this.predicted.has(id) ? json(200, fixture(`${kind}_prediction`)) : error(404, "PREDICTION_NOT_RUN");
    }
    if (sub === "analysis" && method === "POST") {
      this.analysed.add(id);
      return json(200, fixture(`${kind}_analysis`));
    }
    if (sub === "incident" && method === "GET") return json(200, this.incident(id, kind));
    // every other view (analysis, and the Stage 2 views) needs the analysis first
    if (method === "GET") {
      if (!this.analysed.has(id) && sub !== "incident") return error(404, "ANALYSIS_NOT_RUN", "run the analysis for this case first");
      return json(200, fixture(`${kind}_${sub.replace("-", "_")}`));
    }
    throw new Error(`fake API: no route for ${method} ${p}`);
  }
}
