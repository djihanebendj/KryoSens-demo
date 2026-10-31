import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import CopilotPage from "@/app/copilot/page";
import Dashboard from "@/app/dashboard/page";
import PredictionPage from "@/app/prediction/page";
import QualityPage from "@/app/quality/page";
import UploadPage from "@/app/upload/page";
import { CaseProvider } from "@/lib/case-context";
import { formatTimestamp } from "@/lib/format";

import { error, FakeApi, FIXTURE_IDS } from "./fake-api";
import forecastCase from "./fixtures/forecast_case.json";

vi.mock("next/navigation", () => ({ useRouter: () => ({ push: vi.fn() }), usePathname: () => "/" }));

const inCase = (ui: React.ReactNode) => render(<CaseProvider>{ui}</CaseProvider>);

describe("empty, loading, error and not-available states", () => {
  let fake: FakeApi;
  beforeEach(() => {
    fake = new FakeApi().install();
  });

  it("empty: pages without a case point to Upload and make no case request", async () => {
    inCase(<QualityPage />);
    expect(await screen.findByText("No data uploaded yet")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Upload Data" })).toHaveAttribute("href", "/upload");
    expect(fake.calls.filter((c) => c.path.startsWith("/api/v1/cases"))).toHaveLength(0);
  });

  it("empty: dashboard", async () => {
    inCase(<Dashboard />);
    expect(await screen.findByText("No case yet")).toBeInTheDocument();
  });

  it("loading: shown while the quality report is fetched", async () => {
    window.sessionStorage.setItem("coldsense.caseId", FIXTURE_IDS.forecast);
    let release!: () => void;
    const gate = new Promise<void>((r) => (release = r));
    const real = globalThis.fetch;
    globalThis.fetch = (async (...a: Parameters<typeof fetch>) => {
      if (String(a[0]).endsWith("/quality")) await gate;
      return real(...a);
    }) as typeof fetch;
    inCase(<QualityPage />);
    expect(await screen.findByText(/Loading the data-quality report/)).toBeInTheDocument();
    release();
    expect(await screen.findByTestId("quality")).toBeInTheDocument();
  });

  it("error: an expired case is reported and dropped from the shared context", async () => {
    window.sessionStorage.setItem("coldsense.caseId", FIXTURE_IDS.forecast);
    fake.overrides.set(`GET /api/v1/cases/${FIXTURE_IDS.forecast}`, () => error(410, "CASE_EXPIRED", "this case expired"));
    inCase(<QualityPage />);
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveAttribute("data-code", "CASE_EXPIRED");
    expect(alert).toHaveTextContent(/Upload the file again/);
    expect(window.sessionStorage.getItem("coldsense.caseId")).toBeNull();
  });

  it("error: backend unavailable offers a retry", async () => {
    window.sessionStorage.setItem("coldsense.caseId", FIXTURE_IDS.forecast);
    let down = true;
    fake.overrides.set(`GET /api/v1/cases/${FIXTURE_IDS.forecast}/quality`, () =>
      down ? error(502, "BACKEND_UNAVAILABLE", "not reachable", true) : new Response(null, { status: 500 }),
    );
    inCase(<QualityPage />);
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveAttribute("data-code", "BACKEND_UNAVAILABLE");
    down = false;
    fake.overrides.delete(`GET /api/v1/cases/${FIXTURE_IDS.forecast}/quality`);
    await userEvent.click(screen.getByRole("button", { name: "Try again" }));
    expect(await screen.findByTestId("quality")).toBeInTheDocument();
  });

  it("error: an upload error from the API is shown with its code", async () => {
    fake.overrides.set("POST /api/v1/cases", () => error(422, "DATA_UNREADABLE", "the file cannot be interpreted: no temperature column"));
    const user = userEvent.setup();
    inCase(<UploadPage />);
    await user.upload(screen.getByLabelText(/Log file/), new File(["a,b\n1,2\n"], "x.csv"));
    await user.click(screen.getByRole("button", { name: /Upload/ }));
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveAttribute("data-code", "DATA_UNREADABLE");
    expect(window.sessionStorage.getItem("coldsense.caseId")).toBeNull();
  });

  it("client-side checks stop a file that is too large or of the wrong type before it is sent", async () => {
    const user = userEvent.setup({ applyAccept: false });
    inCase(<UploadPage />);
    const big = new File(["x"], "big.csv");
    Object.defineProperty(big, "size", { value: 11 * 1024 * 1024 });
    await user.upload(screen.getByLabelText(/Log file/), big);
    await user.click(screen.getByRole("button", { name: /Upload/ }));
    expect(await screen.findByText(/larger than 10 MiB/)).toBeInTheDocument();
    await user.upload(screen.getByLabelText(/Log file/), new File(["x"], "log.xlsx"));
    await user.click(screen.getByRole("button", { name: /Upload/ }));
    expect(await screen.findByText(/Only .csv, .txt files/)).toBeInTheDocument();
    expect(fake.calls.filter((c) => c.method === "POST")).toHaveLength(0);
  });

  it("prediction is blocked when the data-quality check refused the log", async () => {
    window.sessionStorage.setItem("coldsense.caseId", FIXTURE_IDS.forecast);
    fake.overrides.set(`GET /api/v1/cases/${FIXTURE_IDS.forecast}`, () => {
      const s = structuredClone(forecastCase);
      s.quality.verdict = "REFUSED";
      s.quality.suitable_for_analysis = false;
      return new Response(JSON.stringify(s), { status: 200, headers: { "Content-Type": "application/json" } });
    });
    inCase(<PredictionPage />);
    expect(await screen.findByText(/Prediction is blocked/)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Run prediction/ })).not.toBeInTheDocument();
  });

  it("invalid prediction input is caught before sending", async () => {
    window.sessionStorage.setItem("coldsense.caseId", FIXTURE_IDS.forecast);
    const user = userEvent.setup();
    inCase(<PredictionPage />);
    const h = await screen.findByLabelText(/Horizon/);
    await user.clear(h);
    await user.type(h, "-5");
    await user.click(screen.getByRole("button", { name: /Run prediction/ }));
    expect(await screen.findByText(/positive number of minutes/)).toBeInTheDocument();
    await waitFor(() => expect(fake.calls.some((c) => c.method === "POST")).toBe(false));
  });

  it("copilot page shows the server's FEATURE_DISABLED reason and never calls the copilot endpoint", async () => {
    inCase(<CopilotPage />);
    expect(await screen.findByTestId("copilot-status")).toHaveTextContent("FEATURE_DISABLED");
    expect(screen.getByTestId("copilot-reason")).toHaveTextContent(/Phase 8 acceptance record/);
    expect(fake.calls.some((c) => c.path.endsWith("/copilot"))).toBe(false);
  });
});

describe("formatTimestamp", () => {
  it("keeps naive log time as written and labels aware time as UTC", () => {
    expect(formatTimestamp("2026-10-20T23:27:47.045802")).toBe("2026-10-20 23:27:47");
    expect(formatTimestamp("2026-10-20T23:27:47Z")).toBe("2026-10-20 23:27:47 UTC");
    expect(formatTimestamp("2026-10-20T23:27:47+01:00")).toBe("2026-10-20 22:27:47 UTC");
    expect(formatTimestamp(null)).toBe("—");
  });
});
