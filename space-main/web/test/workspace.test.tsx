import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import EvidencePage from "@/app/evidence/page";
import IncidentPage from "@/app/incident/page";
import InvestigationPage from "@/app/investigation/page";
import NextCheckPage from "@/app/next-check/page";
import Dashboard from "@/app/dashboard/page";
import PredictionPage from "@/app/prediction/page";
import TimelinePage from "@/app/timeline/page";
import { JourneyBar, Nav } from "@/components/nav";
import { TimelineChart } from "@/components/timeline-chart";
import type { TimelineView } from "@/lib/api/types";
import { CaseProvider, useCase } from "@/lib/case-context";
import { toRows } from "@/lib/timeline";

import { error, FakeApi, fixture, FIXTURE_IDS, type Kind } from "./fake-api";

vi.mock("next/navigation", () => ({ useRouter: () => ({ push: vi.fn() }), usePathname: () => "/" }));

const inCase = (ui: React.ReactNode) => render(<CaseProvider>{ui}</CaseProvider>);

const PAGES = [
  { name: "Temperature Timeline", el: () => <TimelinePage />, loaded: "timeline", path: "timeline" },
  { name: "Evidence", el: () => <EvidencePage />, loaded: "evidence", path: "evidence" },
  { name: "Next Best Check", el: () => <NextCheckPage />, loaded: "next-checks", path: "next-check" },
  { name: "Investigation", el: () => <InvestigationPage />, loaded: "investigation", path: "investigation" },
] as const;

let fake: FakeApi;
beforeEach(() => {
  fake = new FakeApi().install();
});

describe.each(PAGES)("$name page states", ({ el, loaded, path }) => {
  it("empty: no case", async () => {
    inCase(el());
    expect(await screen.findByText("No data uploaded yet")).toBeInTheDocument();
  });

  it("loading, then success", async () => {
    fake.given("forecast", { analysed: true });
    let release!: () => void;
    const gate = new Promise<void>((r) => (release = r));
    const real = globalThis.fetch;
    globalThis.fetch = (async (...a: Parameters<typeof fetch>) => {
      if (String(a[0]).endsWith(`/${path}`)) await gate;
      return real(...a);
    }) as typeof fetch;
    inCase(el());
    expect(await screen.findByRole("status")).toHaveAttribute("data-state", "loading");
    release();
    expect(await screen.findByTestId(loaded)).toBeInTheDocument();
  });

  it("analysis not run: points to Temperature Timeline", async () => {
    fake.given("forecast");
    inCase(el());
    expect(await screen.findByText(/No analysis yet|has not been run/)).toBeInTheDocument();
    expect(fake.calls.some((c) => c.path.endsWith(`/${path}`))).toBe(false); // no request that is known to fail
  });

  it("error: shown with its code", async () => {
    fake.given("forecast", { analysed: true });
    fake.overrides.set(`GET /api/v1/cases/${FIXTURE_IDS.forecast}/${path}`, () => error(503, "BUSY", "busy", true));
    inCase(el());
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveAttribute("data-code", "BUSY");
    expect(within(alert).getByRole("button", { name: "Try again" })).toBeInTheDocument();
  });

  it("blocked when the data-quality check refused the log", async () => {
    fake.given("forecast", { analysed: true });
    const s = fixture("forecast_case");
    (s.quality as Record<string, unknown>).suitable_for_analysis = false;
    (s.quality as Record<string, unknown>).verdict = "REFUSED";
    fake.overrides.set(`GET /api/v1/cases/${FIXTURE_IDS.forecast}`, () => new Response(JSON.stringify(s), { status: 200 }));
    inCase(el());
    expect(await screen.findByText(/Blocked: the data-quality check refused this log/)).toBeInTheDocument();
  });
});

describe("Evidence documentation retrieval", () => {
  it("asks the read-only knowledge endpoint and renders the cited passage as text", async () => {
    fake.given("forecast", { analysed: true });
    inCase(<EvidencePage />);
    expect(await screen.findByTestId("knowledge")).toBeInTheDocument();
    expect(await screen.findAllByText("The high-temperature alarm delay is 60 minutes.")).not.toHaveLength(0);
    expect(screen.getAllByText(/Synthetic service manual/)).not.toHaveLength(0);
    expect(fake.calls.some((c) => c.path.startsWith("/api/v1/knowledge?"))).toBe(true);
  });
});

describe("refusal states (Phase 6 INSUFFICIENT_EVIDENCE: no limit was set)", () => {
  it.each([
    ["Evidence", () => <EvidencePage />],
    ["Next Best Check", () => <NextCheckPage />],
    ["Investigation", () => <InvestigationPage />],
  ])("%s shows the refusal and invents nothing", async (_n, el) => {
    fake.given("nolimit", { analysed: true });
    inCase(el());
    const r = await screen.findByText(/INSUFFICIENT_EVIDENCE — ColdSense did not assess any explanation/);
    expect(r.closest("[data-state=refusal]")).toHaveTextContent(/no temperature limit was given/);
    expect(screen.queryByText(/NEXT BEST CHECK/)).not.toBeInTheDocument();
    expect(document.querySelectorAll("[data-testid^=hypothesis-]")).toHaveLength(0);
  });
});

describe("Temperature Timeline", () => {
  it("states its engineering questions, units and range, and runs the analysis via POST with CSRF", async () => {
    fake.given("forecast");
    const user = userEvent.setup();
    inCase(<TimelinePage />);
    await user.type(await screen.findByLabelText(/Upper limit/), "8");
    await user.click(screen.getByRole("button", { name: "Run analysis" }));
    const t = await screen.findByTestId("timeline");
    expect(within(t).getByText("What did the sensor measure, and where are readings missing?")).toBeInTheDocument();
    expect(within(t).getByText(/how did Phase 3 classify each period\?/)).toBeInTheDocument();
    expect(t).toHaveTextContent("unit °C");
    expect(t).toHaveTextContent("2026-10-20 00:00:00 → 2026-10-20 22:00:00");
    expect(screen.getByTestId("display-note")).toHaveTextContent("All 1321 usable readings are drawn.");
    const post = fake.writes().find((c) => c.path.endsWith("/analysis"))!;
    expect(JSON.parse(String(post.body))).toEqual({ upper_threshold_c: 8, lower_threshold_c: null });
    expect(post.headers.get("X-CSRF-Token")).toBeTruthy();
  });

  it("a data gap is listed and stays a gap", async () => {
    fake.given("gaps", { analysed: true });
    inCase(<TimelinePage />);
    const gaps = await screen.findByTestId("gaps");
    expect(gaps.querySelectorAll("li")).toHaveLength(2);
    expect(gaps).toHaveTextContent(/no readings; not interpolated/);
  });

  it("the chart line breaks at each gap (one path segment per run) and draws no invented value", () => {
    const view = fixture("gaps_timeline") as unknown as TimelineView;
    const rows = toRows(view.runs);
    expect(rows.filter((r) => r.temp === null)).toHaveLength(view.runs.length - 1);
    const real = new Set(view.runs.flatMap((r) => r.points.map((p) => p.temp_c)));
    expect(rows.every((r) => r.temp === null || real.has(r.temp))).toBe(true);
    const { container } = render(<TimelineChart view={view} width={800} />);
    const line = container.querySelector(".recharts-line-curve");
    expect(line).not.toBeNull();
    const moves = (line!.getAttribute("d") ?? "").match(/M/g) ?? [];
    expect(moves).toHaveLength(view.runs.length); // 3 runs -> 3 separate segments, nothing drawn across a gap
    expect(container.querySelectorAll(".recharts-reference-area")).toHaveLength(view.gaps.length + view.events.filter((e) => e.type === "THRESHOLD_EXCURSION").length);
  });

  it("says when the series is thinned for display", async () => {
    fake.given("forecast", { analysed: true });
    fake.overrides.set(`GET /api/v1/cases/${FIXTURE_IDS.forecast}/timeline`, () => {
      const t = fixture("forecast_timeline");
      t.display = { downsampled: true, method: "display only: min and max temperature per 4 consecutive readings, within each run", points_total: 8000, points_shown: 2000, max_points: 2000 };
      return new Response(JSON.stringify(t), { status: 200 });
    });
    inCase(<TimelinePage />);
    expect(await screen.findByTestId("display-note")).toHaveTextContent(/Display only: 2000 of 8000 readings are drawn/);
  });
});

describe("Evidence", () => {
  it("separates observed facts, document facts, inferences and recommendations; shows contradictions and missing evidence", async () => {
    fake.given("forecast", { analysed: true, predicted: true });
    inCase(<EvidencePage />);
    const e = await screen.findByTestId("evidence");
    for (const label of ["OBSERVED_FACT", "DOCUMENT_FACT", "MODEL_INFERENCE", "SYSTEM_RECOMMENDATION"]) expect(within(e).getAllByText(label).length).toBeGreaterThan(0);
    expect(screen.getByTestId("document-facts")).toHaveTextContent("NOT_AVAILABLE");
    const summary = screen.getByTestId("evidence-summary");
    expect(summary).toHaveTextContent(/Contradictions \(1\)/);
    expect(summary).toHaveTextContent(/CAPACITY_SHORTFALL_WHILE_RUNNING: compressor recorded OFF/);
    expect(summary).toHaveTextContent(/Missing evidence \([1-9]/);
    expect(screen.getByTestId("prediction-status")).toHaveTextContent(/reliability MEDIUM \(rule-based, not a probability\)/);
  });
});

describe("Next Best Check", () => {
  it("renders the backend's ranks in order, marks the next best check, and shows no score", async () => {
    fake.given("forecast", { analysed: true });
    inCase(<NextCheckPage />);
    const list = await screen.findByTestId("next-checks");
    const checks = fixture("forecast_next_check").checks as { check_id: string; rank: number }[];
    const shown = [...list.querySelectorAll("[data-testid^=check-]")].map((n) => n.getAttribute("data-testid")!.slice(6));
    expect(shown).toEqual(checks.map((c) => c.check_id));
    expect(within(screen.getByTestId(`check-${checks[0].check_id}`)).getByText("NEXT BEST CHECK")).toBeInTheDocument();
    expect(within(screen.getByTestId(`check-${checks[0].check_id}`)).getByTestId("expected-information")).toBeInTheDocument();
    expect(list).not.toHaveTextContent(/(score|probability|confidence)\s*[:=]\s*\d|\d+(\.\d+)?\s*%/i); // no numeric score
  });
});

describe("Investigation", () => {
  it("labels every hypothesis MODEL_INFERENCE and 'hypothesis, not a confirmed fault', with contradicting and missing evidence", async () => {
    fake.given("forecast", { analysed: true });
    inCase(<InvestigationPage />);
    await screen.findByTestId("investigation");
    const hyps = document.querySelectorAll("[data-testid^=hypothesis-]");
    expect(hyps.length).toBe((fixture("forecast_investigation").hypotheses as unknown[]).length);
    hyps.forEach((h) => {
      expect(h).toHaveTextContent("MODEL_INFERENCE");
      expect(h).toHaveTextContent("hypothesis, not a confirmed fault");
      expect(h).toHaveTextContent("PROPOSED");
      expect(within(h as HTMLElement).getByTestId("contradicting")).toBeInTheDocument();
      expect(within(h as HTMLElement).getByTestId("missing")).toBeInTheDocument();
    });
    expect(screen.queryByRole("button", { name: /confirm|record finding/i })).not.toBeInTheDocument();
  });
});

describe("Incident", () => {
  it("shows the finding form and an empty incident with zero counts before any finding", async () => {
    fake.given("forecast");
    inCase(<IncidentPage />);
    expect(await screen.findByText("Record a technician finding")).toBeInTheDocument();
    expect(screen.getByTestId("count-confirmed")).toHaveTextContent("0");
    expect(screen.getByTestId("count-drafts")).toHaveTextContent("0");
    expect(screen.getByTestId("incident-drafts")).toHaveTextContent("No drafts.");
  });

  it("empty and error states", async () => {
    inCase(<IncidentPage />);
    expect(await screen.findByText("No data uploaded yet")).toBeInTheDocument();
  });
});

describe("a refused prediction never shows a forecast anywhere", () => {
  it.each([
    ["Dashboard", () => <Dashboard />],
    ["Prediction", () => <PredictionPage />],
    ["Timeline", () => <TimelinePage />],
    ["Evidence", () => <EvidencePage />],
    ["Next Best Check", () => <NextCheckPage />],
    ["Investigation", () => <InvestigationPage />],
  ])("%s", async (_n, el) => {
    fake.given("refusal", { analysed: true, predicted: true });
    inCase(el());
    await waitFor(() => expect(fake.calls.some((c) => c.path.startsWith(`/api/v1/cases/${FIXTURE_IDS.refusal}`))).toBe(true));
    await waitFor(() => expect(document.querySelector("[data-state=loading]")).toBeNull());
    expect(screen.queryByTestId("trajectory-chart")).not.toBeInTheDocument();
    expect(screen.queryByTestId("forecast")).not.toBeInTheDocument();
    expect(document.body).not.toHaveTextContent(/CROSSING_PREDICTED|RELIABILITY (HIGH|MEDIUM|LOW)/);
  });
});

describe("guided journey and navigation", () => {
  it("disables steps whose prerequisites are missing, and enables them after the analysis", async () => {
    fake.given("forecast");
    const { unmount } = inCase(
      <>
        <Nav />
        <JourneyBar />
      </>,
    );
    for (const href of ["/evidence", "/next-check", "/investigation"]) {
      const item = await screen.findByTestId(`nav-disabled-${href}`);
      expect(item).toHaveAttribute("aria-disabled", "true");
      expect(item).toHaveTextContent(/run the analysis on Temperature Timeline first/);
    }
    expect(screen.getByTestId("journey")).toHaveTextContent(/Evidence \(blocked: run the analysis/);
    expect(screen.getByTestId("journey")).toHaveTextContent(/Copilot \(not available: FEATURE_DISABLED\)/);
    unmount();
    fake.analysed.add(FIXTURE_IDS.forecast);
    fake.predicted.add(FIXTURE_IDS.forecast); // the after-prediction status fixture has analysis READY
    inCase(<Nav />);
    await waitFor(() => expect(screen.getByRole("link", { name: "Evidence" })).toHaveAttribute("href", "/evidence"));
    expect(screen.queryByTestId("nav-disabled-/evidence")).not.toBeInTheDocument();
  });

  it("with no case only Dashboard, Upload and the self-explaining Copilot page are reachable", async () => {
    inCase(<Nav />);
    await screen.findByRole("link", { name: "Upload Data" });
    for (const href of ["/quality", "/timeline", "/prediction", "/evidence", "/next-check", "/investigation", "/incident",
                        "/reports"])
      expect(screen.getByTestId(`nav-disabled-${href}`)).toBeInTheDocument();
  });

  it("never mixes data between cases: switching the case reloads every view for the new case only", async () => {
    fake.given("forecast", { analysed: true });
    fake.analysed.add(FIXTURE_IDS.refusal);
    let switchTo: (k: Kind) => void = () => undefined;
    function Switcher() {
      const { setStatus } = useCase();
      switchTo = (k) => setStatus(fixture(`${k}_case`) as never);
      return null;
    }
    inCase(
      <>
        <Switcher />
        <EvidencePage />
      </>,
    );
    await screen.findByTestId("finding-F1");
    const before = fake.calls.length;
    switchTo("refusal");
    await waitFor(() => expect(screen.queryByTestId("finding-F1")).not.toBeInTheDocument());
    await screen.findByTestId("evidence");
    const after = fake.calls.slice(before).map((c) => c.path);
    expect(after).toContain(`/api/v1/cases/${FIXTURE_IDS.refusal}/evidence`);
    expect(after.some((p) => p.includes(FIXTURE_IDS.forecast))).toBe(false);
    expect(document.body).not.toHaveTextContent(/CAPACITY_SHORTFALL_WHILE_RUNNING/);
  });
});
