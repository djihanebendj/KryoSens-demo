import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import IncidentPage from "@/app/incident/page";
import ReportsPage from "@/app/reports/page";
import { CaseProvider } from "@/lib/case-context";

import { FakeApi } from "./fake-api";

vi.mock("next/navigation", () => ({ useRouter: () => ({ push: vi.fn() }), usePathname: () => "/" }));

const inCase = (ui: React.ReactNode) => render(<CaseProvider>{ui}</CaseProvider>);

describe("technician findings and reports", () => {
  let fake: FakeApi;
  beforeEach(() => {
    fake = new FakeApi().install();
    fake.given("forecast");
  });

  async function prepare(statement: string, window = "") {
    const user = userEvent.setup();
    await screen.findByText("Record a technician finding");
    const box = screen.getByLabelText(/Statement/);
    await user.type(box, statement);
    if (window) await user.type(screen.getByLabelText(/Event window/), window);
    await user.click(screen.getByRole("button", { name: "Prepare draft" }));
    const panel = await screen.findByTestId("finding-draft");
    return { user, panel };
  }

  it("shows the exact backend draft; cancel writes nothing and marks it cancelled", async () => {
    inCase(<IncidentPage />);
    const statement = "Door seal worn.\n<script>alert('x')</script>  ";
    const { user, panel } = await prepare(statement, "overnight");

    // The exact backend statement, including whitespace and markup-looking text, is text, never HTML.
    expect(screen.getByTestId("finding-draft-statement").textContent).toBe(statement);
    expect(panel.querySelector("script")).toBeNull();
    expect(panel).toHaveTextContent("DRAFT · NOT HISTORY");
    expect(panel).toHaveTextContent("Content hash");

    await user.click(screen.getByRole("button", { name: "Cancel draft" }));
    expect(await screen.findByText(/Draft cancelled\. Nothing was written/)).toBeInTheDocument();
    await waitFor(() => expect(screen.getByText("cancelled — nothing was recorded")).toBeInTheDocument());

    expect(fake.calls.some((c) => c.method === "POST" && c.path.endsWith("/confirm"))).toBe(false);
    expect(fake.calls.some((c) => c.method === "POST" && c.path.endsWith("/cancel"))).toBe(true);
    expect(screen.getByTestId("count-confirmed")).toHaveTextContent("0");
    expect(screen.getByTestId("count-drafts")).toHaveTextContent("1");
    // The cancelled statement is shown as a draft, not as a confirmed finding.
    expect(screen.getByTestId("incident-drafts")).toHaveTextContent("Door seal worn.");
    expect(screen.getByTestId("incident-drafts")).toHaveTextContent("<script>alert('x')</script>");
    expect(screen.getByTestId("incident-confirmed")).not.toHaveTextContent("Door seal worn.");
  });

  it("confirm sends the single-use token once and shows a confirmed record with its outcome", async () => {
    inCase(<IncidentPage />);
    const statement = "The compressor restarted after the power cut.";
    const { user } = await prepare(statement);
    await user.selectOptions(screen.getByLabelText(/Outcome/), "RESOLVED");
    await user.click(screen.getByRole("button", { name: "Confirm finding" }));

    expect(await screen.findByText(/Finding finding-0001 confirmed/)).toBeInTheDocument();
    const confirmCall = fake.calls.find((c) => c.method === "POST" && c.path.endsWith("/confirm"));
    expect(confirmCall).toBeDefined();
    const body = JSON.parse(String(confirmCall!.body)) as { token: string; outcome: string };
    expect(body.token).toMatch(/^test-token-draft-0001$/);
    expect(body.outcome).toBe("RESOLVED");

    await waitFor(() => expect(screen.getByTestId("count-confirmed")).toHaveTextContent("1"));
    expect(screen.getByTestId("count-verified")).toHaveTextContent("1");
    const confirmed = screen.getByTestId("outcome-verified");
    expect(confirmed).toHaveTextContent(statement);
    expect(confirmed).toHaveTextContent("RESOLVED");
    expect(confirmed).toHaveTextContent("Confirmed by you in this web application");
    expect(screen.queryByTestId("finding-draft")).not.toBeInTheDocument();
  });

  it("a double click on Confirm sends exactly one confirmation and writes one record", async () => {
    inCase(<IncidentPage />);
    const { user } = await prepare("Double-click statement.");
    await user.dblClick(screen.getByRole("button", { name: "Confirm finding" }));
    expect(await screen.findByText(/Finding finding-0001 confirmed/)).toBeInTheDocument();
    expect(fake.calls.filter((c) => c.method === "POST" && c.path.endsWith("/confirm"))).toHaveLength(1);
    await waitFor(() => expect(screen.getByTestId("count-confirmed")).toHaveTextContent("1"));
  });

  it("shows a confirmed record and an excluded draft distinctly in the report", async () => {
    const first = inCase(<IncidentPage />);
    const confirmedStatement = "The door seal was worn.";
    {
      const { user } = await prepare(confirmedStatement);
      await user.selectOptions(screen.getByLabelText(/Outcome/), "RESOLVED");
      await user.click(screen.getByRole("button", { name: "Confirm finding" }));
      await screen.findByText(/Finding finding-0001 confirmed/);
    }
    const draftStatement = "A second technician will check the heater.";
    {
      const { user } = await prepare(draftStatement);
      expect(await screen.findByTestId("finding-draft")).toHaveTextContent(draftStatement);
      void user;
    }
    first.unmount();

    inCase(<ReportsPage />);
    const user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: "Generate report" }));
    const text = await screen.findByTestId("report-text");
    expect(text).toHaveTextContent(confirmedStatement);
    expect(text).not.toHaveTextContent(draftStatement);
    expect(screen.getByTestId("report-confirmed")).toHaveTextContent("CONFIRMED — included");
    expect(screen.getByTestId("report-drafts")).toHaveTextContent("DRAFT — excluded from history");
    expect(screen.getByTestId("report-drafts")).toHaveTextContent(draftStatement);
    expect(screen.getByTestId("report-meta")).toHaveTextContent("PASSED");
    expect(screen.getByTestId("report-meta")).toHaveTextContent("1 draft(s) excluded");
  }, 10_000);

  it("a refused report shows the violations and no report text", async () => {
    fake.refuseReport = true;
    inCase(<ReportsPage />);
    const user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: "Generate report" }));
    const refused = await screen.findByTestId("report-refused");
    expect(refused).toHaveTextContent("REPORT REFUSED");
    expect(refused).toHaveTextContent("UNSUPPORTED_NUMBER");
    expect(screen.queryByTestId("report-text")).not.toBeInTheDocument();
  });
});
