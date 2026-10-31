import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { FREE_TEXT_NOTICE, GuidedStart, labelFor, SAFETY_NOTICE } from "@/app/copilot/guided-start";
import CopilotPage from "@/app/copilot/page";
import { CaseProvider } from "@/lib/case-context";

import { FakeApi, fixture, FIXTURE_IDS } from "./fake-api";

vi.mock("next/navigation", () => ({ useRouter: () => ({ push: vi.fn() }), usePathname: () => "/copilot" }));

const inCase = (ui: React.ReactNode) => render(<CaseProvider>{ui}</CaseProvider>);
const json = (body: unknown) => new Response(JSON.stringify(body), { status: 200, headers: { "Content-Type": "application/json" } });

let fake: FakeApi;
beforeEach(() => {
  fake = new FakeApi().install();
});

describe("Copilot page notices", () => {
  it("shows the fixed free-text and safety notices and only GETs, never the copilot endpoint", async () => {
    fake.given("forecast", { analysed: true });
    inCase(<CopilotPage />);
    expect(await screen.findByTestId("guided-start")).toBeInTheDocument();
    expect(screen.getByTestId("free-text-notice")).toHaveTextContent(FREE_TEXT_NOTICE);
    expect(screen.getByTestId("safety-notice")).toHaveTextContent(SAFETY_NOTICE);
    expect(fake.calls.some((c) => c.path.includes("/copilot"))).toBe(false);
    expect(fake.calls.filter((c) => c.method !== "GET")).toEqual([]);
    expect(document.querySelector("textarea, input[type=text]")).toBeNull();
  });
});

describe("Guided Start", () => {
  it("blocked: analysis not run uses the shared prerequisite state and requests nothing", async () => {
    fake.given("forecast");
    inCase(<GuidedStart />);
    expect(await screen.findByText(/has not been run/)).toBeInTheDocument();
    expect(fake.calls.some((c) => /\/(next-check|investigation|evidence)$/.test(c.path))).toBe(false);
  });

  it("blocked: a refused data-quality check shows the shared blocked state", async () => {
    fake.given("forecast", { analysed: true });
    const s = fixture("forecast_case");
    (s.quality as Record<string, unknown>).suitable_for_analysis = false;
    (s.quality as Record<string, unknown>).verdict = "REFUSED";
    fake.overrides.set(`GET /api/v1/cases/${FIXTURE_IDS.forecast}`, () => json(s));
    inCase(<GuidedStart />);
    expect(await screen.findByText(/Blocked: the data-quality check refused this log/)).toBeInTheDocument();
  });

  it("diagnosis refusal: only the backend's reason, no hypotheses or checks", async () => {
    fake.given("nolimit", { analysed: true });
    inCase(<GuidedStart />);
    const r = await screen.findByTestId("guided-refusal");
    expect(r).toHaveTextContent(/no temperature limit was given/);
    expect(screen.queryByTestId("where-to-start")).toBeNull();
    expect(document.querySelectorAll("[data-testid^=guided-hypothesis-]")).toHaveLength(0);
  });

  it("prediction refusal: only the backend's reason, no trajectory, interval or reliability", async () => {
    fake.given("refusal", { analysed: true });
    inCase(<GuidedStart />);
    const r = await screen.findByTestId("guided-refusal");
    expect(r).toHaveTextContent("there is no evidence of any period without active cooling; the physical models need one");
    expect(screen.queryByTestId("leading-hypotheses")).toBeNull();
    expect(document.body.textContent).not.toMatch(/trajectory|interval|reliability/i);
  });

  it("normal: renders the backend's top check, its reason and expected information, and PROPOSED hypotheses", async () => {
    fake.given("forecast", { analysed: true });
    const next = fixture("forecast_next_check") as { checks: { check_id: string; why_useful: string; expected_information: string[] }[] };
    const inv = fixture("forecast_investigation") as { hypotheses: { id: string; statement: string }[] };
    inCase(<GuidedStart />);
    const start = await screen.findByTestId("where-to-start");
    expect(start).toHaveTextContent("Review the site's operational records");
    expect(start).toHaveTextContent(next.checks[0].why_useful);
    expect(screen.getByTestId("expected-information")).toHaveTextContent(next.checks[0].expected_information[0]);
    expect(document.querySelectorAll("[data-testid^=guided-hypothesis-]")).toHaveLength(inv.hypotheses.length);
    expect(screen.getByTestId(`guided-hypothesis-${inv.hypotheses[0].id}`)).toHaveTextContent("Proposed hypothesis");
    expect(screen.getByTestId("provenance-line")).toHaveTextContent(/SIMULATED/);
  });

  it("never shows a hypothesis as confirmed", async () => {
    fake.given("forecast", { analysed: true });
    inCase(<GuidedStart />);
    await screen.findByTestId("leading-hypotheses");
    const text = document.body.textContent ?? "";
    const confirmed = text.match(/.{0,6}confirmed/gi) ?? [];
    expect(confirmed.length).toBeGreaterThan(0);                         // the backend's note is present...
    expect(confirmed.every((m) => /not a confirmed/i.test(m))).toBe(true); // ...and only ever as "not a confirmed"
  });

  it("unknown enum code falls back to the raw code", async () => {
    expect(labelFor("SOME_NEW_BACKEND_CODE")).toBe("SOME_NEW_BACKEND_CODE");
    fake.given("forecast", { analysed: true });
    const next = fixture("forecast_next_check") as { next_best_check: string; checks: { check_id: string }[] };
    next.checks[0].check_id = "SOME_NEW_BACKEND_CODE";
    next.next_best_check = "SOME_NEW_BACKEND_CODE";
    fake.overrides.set(`GET /api/v1/cases/${FIXTURE_IDS.forecast}/next-check`, () => json(next));
    inCase(<GuidedStart />);
    const start = await screen.findByTestId("where-to-start");
    expect(start.querySelector(".font-medium")).toHaveTextContent("SOME_NEW_BACKEND_CODE");
  });
});
