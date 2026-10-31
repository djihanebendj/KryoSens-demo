import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import Dashboard from "@/app/dashboard/page";
import PredictionPage from "@/app/prediction/page";
import QualityPage from "@/app/quality/page";
import UploadPage from "@/app/upload/page";
import { CaseProvider } from "@/lib/case-context";

import { CSRF_TOKEN, FakeApi, FIXTURE_IDS } from "./fake-api";

const push = vi.fn();
vi.mock("next/navigation", () => ({ useRouter: () => ({ push }), usePathname: () => "/" }));

function inCase(ui: React.ReactNode) {
  return render(<CaseProvider>{ui}</CaseProvider>);
}

async function useSample(name: RegExp) {
  const user = userEvent.setup();
  const view = inCase(<UploadPage />);
  await user.click(await screen.findByRole("button", { name }));
  await waitFor(() => expect(push).toHaveBeenCalledWith("/quality"));
  view.unmount();
}

async function predict(threshold: string, horizon = "120") {
  const user = userEvent.setup();
  inCase(<PredictionPage />);
  const h = await screen.findByLabelText(/Horizon/);
  await user.clear(h);
  await user.type(h, horizon);
  await user.type(screen.getByLabelText(/Threshold/), threshold);
  await user.click(screen.getByRole("button", { name: /Run prediction/ }));
}

describe("upload -> quality -> prediction", () => {
  let fake: FakeApi;
  beforeEach(() => {
    push.mockReset();
    fake = new FakeApi().install();
  });

  it("runs the whole flow on one shared case (bundled SIMULATED sample) and shows a forecast with baselines", async () => {
    await useSample(/SIMULATED_power_cut_22h_1min/);
    const post = fake.calls.find((c) => c.method === "POST")!;
    expect(post.path).toBe("/api/v1/cases/samples/power-cut-22h");
    expect(post.headers.get("X-CSRF-Token")).toBe(CSRF_TOKEN);
    expect(window.sessionStorage.getItem("coldsense.caseId")).toBe(FIXTURE_IDS.forecast);

    const quality = inCase(<QualityPage />);
    expect(await screen.findByTestId("quality")).toBeInTheDocument();
    expect(screen.getByTestId("provenance-line")).toHaveTextContent(/SIMULATED.*bundled ColdSense sample.*SIMULATED_ONLY/);
    expect(screen.getByRole("link", { name: /Continue to Prediction/ })).toHaveAttribute("href", "/prediction");
    quality.unmount();

    await predict("20");
    const forecast = await screen.findByTestId("forecast");
    expect(within(forecast).getByTestId("crossing")).toHaveTextContent("CROSSING_PREDICTED");
    expect(within(forecast).getByText(/rule-based classification, not a probability/)).toBeInTheDocument();
    expect(screen.getByTestId("trajectory-chart")).toBeInTheDocument();
    expect(within(screen.getByTestId("baselines")).getByText("persistence")).toBeInTheDocument();
    expect(screen.queryByTestId("refusal")).not.toBeInTheDocument();
    const sent = fake.calls.find((c) => c.method === "POST" && c.path.endsWith("/prediction"))!;
    expect(sent.headers.get("X-CSRF-Token")).toBe(CSRF_TOKEN);
    expect(JSON.parse(String(sent.body))).toMatchObject({ horizon_min: 120, threshold_c: 20, assess_reliability: true });
    // naive log timestamps are shown as written, never shifted into the browser's time zone
    expect(within(forecast).getByTestId("crossing")).toHaveTextContent("2026-10-20 23:27:47");
  });

  it("an uploaded file is UNVALIDATED and its declaration is shown as not verified", async () => {
    const user = userEvent.setup();
    const view = inCase(<UploadPage />);
    await user.upload(screen.getByLabelText(/Log file/), new File(["ts,temp_c\n1,2\n"], "site_log.csv", { type: "text/csv" }));
    await user.click(screen.getByLabelText(/^REAL/));
    await user.click(screen.getByRole("button", { name: /Upload and check data quality/ }));
    await waitFor(() => expect(push).toHaveBeenCalledWith("/quality"));
    const post = fake.calls.find((c) => c.method === "POST" && c.path.startsWith("/api/v1/cases?"))!;
    expect(post.path).toContain("declared_evidence=REAL");
    expect(post.path).not.toMatch(/evidence=(SIMULATED|UNVALIDATED)|trust/);
    expect(post.body).toBeInstanceOf(File); // raw body, not multipart
    view.unmount();
    inCase(<QualityPage />);
    const line = await screen.findByTestId("provenance-line");
    expect(line).toHaveTextContent(/Analysed as\s*UNVALIDATED/);
    expect(line).toHaveTextContent("UNVALIDATED_PROVENANCE");
    expect(within(line).getByTestId("declaration")).toHaveTextContent("declared by uploader, not verified");
  });

  it("shows a refusal and never a trajectory", async () => {
    await useSample(/SIMULATED_coldroom_24h_1min/);
    await predict("8", "60");
    const refusal = await screen.findByTestId("refusal");
    expect(within(refusal).getByText("MISSING_REQUIRED_INPUT")).toBeInTheDocument();
    expect(within(refusal).getByText(/Needed to retry/)).toBeInTheDocument();
    expect(screen.queryByTestId("trajectory-chart")).not.toBeInTheDocument();
    expect(screen.queryByTestId("forecast")).not.toBeInTheDocument();
    expect(screen.queryByTestId("baselines")).not.toBeInTheDocument();
    expect(document.querySelector("svg polyline")).toBeNull();
    expect(screen.queryByText(/CROSSING_PREDICTED/)).not.toBeInTheDocument();
    expect(screen.queryByText(/RELIABILITY/)).not.toBeInTheDocument();
  });

  it("does not render a trajectory even if a refusal response smuggles one in", async () => {
    fake.tamperPrediction = (p) => ({
      ...p,
      trajectory: [{ ts: "2026-10-21T00:00:00", minutes: 1, temp_c: 99.5, provenance: "PREDICTED" }],
      crossing: { status: "CROSSING_PREDICTED", time: "2026-10-21T00:01:00", minutes: 1 },
    });
    await useSample(/SIMULATED_coldroom_24h_1min/);
    await predict("8", "60");
    await screen.findByTestId("refusal");
    expect(screen.queryByTestId("trajectory-chart")).not.toBeInTheDocument();
    expect(screen.queryByText(/99\.5/)).not.toBeInTheDocument();
    expect(screen.queryByText(/CROSSING_PREDICTED/)).not.toBeInTheDocument();
  });

  it("the dashboard shares the case with the other pages", async () => {
    await useSample(/SIMULATED_power_cut_22h_1min/);
    inCase(<Dashboard />);
    const card = await screen.findByTestId("case-card");
    expect(within(card).getByText("SIMULATED_power_cut_22h_1min.csv")).toBeInTheDocument();
    expect(await screen.findByTestId("copilot-state")).toHaveTextContent("FEATURE_DISABLED");
  });

  it("a rejected CSRF token is refreshed once, then the write is retried", async () => {
    let first = true;
    fake.overrides.set("POST /api/v1/cases/samples/power-cut-22h", () => {
      if (first) {
        first = false;
        return new Response(JSON.stringify({ error: { code: "CSRF_TOKEN_INVALID", message: "x", retryable: false } }), { status: 403 });
      }
      fake.overrides.delete("POST /api/v1/cases/samples/power-cut-22h");
      return globalThis.fetch("/api/v1/cases/samples/power-cut-22h", { method: "POST", headers: { "X-CSRF-Token": CSRF_TOKEN } });
    });
    await useSample(/SIMULATED_power_cut_22h_1min/);
    expect(fake.calls.filter((c) => c.path === "/api/v1/csrf")).toHaveLength(2);
  });
});
