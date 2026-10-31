// Local end-to-end check of Phase 9 through a real browser:
// Next.js (same-origin proxy) -> FastAPI -> accepted Phase 2-6 code, SIMULATED samples only. No external network.
//
//   BASE_URL=http://127.0.0.1:3000 node e2e/local-e2e.cjs [screenshot-dir]
//
// Needs Playwright with Chromium (`require("playwright")`); it is not a project dependency.
const path = require("node:path");
const { chromium, request } = require("playwright");

const BASE = process.env.BASE_URL || "http://127.0.0.1:3000";
const SAMPLES = path.resolve(__dirname, "../../data/sample");
const FORECAST_CSV = path.join(SAMPLES, "SIMULATED_power_cut_22h_1min.csv");
const SHOTS = process.argv[2];
const STAGE2 = process.env.E2E_STAGE2 !== "0";
const results = [];

function check(name, ok, detail = "") {
  results.push({ name, ok: Boolean(ok), detail });
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}${detail ? `  (${detail})` : ""}`);
}

async function useSample(page, filename) {
  await page.goto(`${BASE}/upload`);
  await page.getByRole("button", { name: `Use ${filename}` }).click();
  await page.waitForURL(`${BASE}/quality`);
  await page.getByTestId("quality").waitFor();
}

async function uploadFile(page, file, declared) {
  await page.goto(`${BASE}/upload`);
  await page.setInputFiles("#file", file);
  if (declared) await page.getByLabel(new RegExp(`^${declared}`)).check();
  await page.getByRole("button", { name: /Upload and check data quality/ }).click();
  await page.waitForURL(`${BASE}/quality`);
  await page.getByTestId("quality").waitFor();
}

async function predict(page, horizon, threshold) {
  await page.goto(`${BASE}/prediction`);
  await page.locator("#horizon").fill(String(horizon));
  await page.locator("#threshold").fill(String(threshold));
  await page.getByRole("button", { name: /Run prediction/ }).click();
  await page.locator('[data-testid="forecast"], [data-testid="refusal"]').first().waitFor({ timeout: 60_000 });
}

async function analyse(page, upper) {
  await page.goto(`${BASE}/timeline`);
  await page.locator("#upper-threshold").fill(String(upper));
  await page.getByRole("button", { name: /Run analysis/ }).click();
  await page.getByTestId("timeline").waitFor({ timeout: 60_000 });
}

async function shot(page, name) {
  if (SHOTS) await page.screenshot({ path: path.join(SHOTS, `${name}.png`), fullPage: true });
}

async function pageFetch(page, url, init = {}) {
  return page.evaluate(
    async ([u, i]) => {
      const t = await (await fetch("/api/v1/csrf")).json();
      const headers = { ...(i.headers || {}), ...(i.noToken ? {} : { "X-CSRF-Token": t.csrf_token }) };
      const r = await fetch(u, { method: i.method || "GET", headers, body: i.body });
      return { status: r.status, body: r.status === 204 ? null : await r.json() };
    },
    [url, init],
  );
}

(async () => {
  const browser = await chromium.launch();
  try {
    const a = await browser.newContext();
    const pa = await a.newPage();
    const errors = [];
    pa.on("console", (m) => m.type() === "error" && errors.push(m.text()));

    const home = await pa.goto(`${BASE}/dashboard`);   // "/" is the landing page; the dashboard moved
    const h = home.headers();
    check("security headers on pages", h["content-security-policy"]?.includes("frame-ancestors 'none'") && h["x-frame-options"] === "DENY");
    check("dashboard empty state", await pa.getByText("No case yet").isVisible());
    const nav = await pa.getByRole("navigation", { name: "Main" }).innerText();
    const sections = ["Dashboard", "Upload Data", "Data Quality", "Temperature Timeline", "Prediction", "Evidence", "Next Best Check", "Technician Copilot", "Investigation", "Incident", "Reports"];
    check("navigation shows the spec's 11 sections", sections.every((s) => nav.includes(s)));

    // 1. bundled SIMULATED sample: forecast
    await useSample(pa, "SIMULATED_power_cut_22h_1min.csv");
    check("bundled sample analysed as SIMULATED, scope SIMULATED_ONLY", /SIMULATED[\s\S]*SIMULATED_ONLY/.test(await pa.getByTestId("provenance-line").first().innerText()));
    await shot(pa, "1-quality");
    const sess = (await a.cookies()).find((c) => c.name === "cs_session");
    check("session cookie HttpOnly, SameSite=Lax, opaque id only", sess && sess.httpOnly && sess.sameSite === "Lax" && sess.value.split(".").length === 3);
    const caseA = await pa.evaluate(() => sessionStorage.getItem("coldsense.caseId"));
    if (STAGE2) {
      await analyse(pa, 8);
      check("timeline shows measurements and states its question", (await pa.getByTestId("timeline").innerText()).includes("What did the sensor measure"));
      await shot(pa, "2-timeline");
    }
    await predict(pa, 120, 20);
    check("forecast rendered with chart and baselines", (await pa.getByTestId("trajectory-chart").count()) === 1 && (await pa.getByTestId("baselines").count()) === 1);
    check("crossing predicted", (await pa.getByTestId("crossing").textContent()).includes("CROSSING_PREDICTED"));
    await shot(pa, "3-forecast");
    if (STAGE2) {
      await pa.goto(`${BASE}/evidence`);
      await pa.getByTestId("evidence").waitFor();
      const ev = await pa.getByTestId("evidence").innerText();
      check("evidence separates observed facts, inferences, recommendations", ev.includes("OBSERVED_FACT") && ev.includes("MODEL_INFERENCE") && ev.includes("SYSTEM_RECOMMENDATION"));
      check("evidence shows contradictions and missing evidence", ev.includes("Contradicting") && ev.includes("Missing evidence"));
      await shot(pa, "4-evidence");
      await pa.goto(`${BASE}/next-check`);
      await pa.getByTestId("next-checks").waitFor();
      check("next best check ranked by the backend", (await pa.getByTestId("next-checks").innerText()).includes("REVIEW_OPERATIONAL_RECORDS"));
      await shot(pa, "5-next-check");
      await pa.goto(`${BASE}/investigation`);
      await pa.getByTestId("investigation").waitFor();
      check("investigation labels hypotheses, not confirmed faults", (await pa.getByTestId("investigation").innerText()).includes("hypothesis, not a confirmed fault"));
      await shot(pa, "6-investigation");
      await pa.goto(`${BASE}/incident`);
      await pa.getByTestId("incident-counts").waitFor();
      check("incident shows the stored record (no confirmed finding yet)", (await pa.getByTestId("count-confirmed").innerText()).includes("0"));
      await pa.goto(`${BASE}/copilot`);
      await pa.getByTestId("where-to-start").waitFor();
      check("copilot: free-text disabled notice and rule-based Guided Start", (await pa.getByTestId("free-text-notice").innerText()).includes("Free-text assistant: disabled in this build") && (await pa.getByTestId("where-to-start").innerText()).includes("REVIEW_OPERATIONAL_RECORDS"));
    }

    // 2. bundled sample without compressor state: refusal, never a forecast
    await useSample(pa, "SIMULATED_coldroom_24h_1min.csv");
    await predict(pa, 60, 8);
    check("refusal rendered without a trajectory", (await pa.getByTestId("refusal").count()) === 1 && (await pa.getByTestId("trajectory-chart").count()) === 0);
    await shot(pa, "7-refusal");
    if (STAGE2) {
      let leaked = false;
      for (const p of ["/dashboard", "/timeline", "/evidence", "/next-check", "/investigation"]) {
        await pa.goto(`${BASE}${p}`);
        await pa.waitForLoadState("networkidle");
        leaked ||= (await pa.getByTestId("trajectory-chart").count()) > 0 || (await pa.getByText("CROSSING_PREDICTED").count()) > 0;
      }
      check("refused case shows no forecast on any page", !leaked);
    }

    // 3. anonymous upload declared REAL: analysed as UNVALIDATED, declaration shown as not verified
    await uploadFile(pa, FORECAST_CSV, "REAL");
    const line = await pa.getByTestId("provenance-line").first().innerText();
    check("upload analysed as UNVALIDATED despite declaring REAL", line.includes("UNVALIDATED") && line.includes("declared by uploader, not verified"), line);
    check("no browser console errors in the UI flows", errors.length === 0, errors.join(" | "));

    // 4. CSRF through the real proxy
    const noToken = await pageFetch(pa, `/api/v1/cases/${caseA}`, { method: "DELETE", noToken: true });
    check("write without CSRF token rejected", noToken.status === 403 && noToken.body.error.code === "CSRF_TOKEN_MISSING", JSON.stringify(noToken));
    const api = await request.newContext({ baseURL: BASE, storageState: await a.storageState() });
    const tok = (await (await api.get("/api/v1/csrf")).json()).csrf_token;
    const cross = await api.delete(`/api/v1/cases/${caseA}`, { headers: { Origin: "https://attacker.example", "X-CSRF-Token": tok } });
    check("cross-site Origin rejected", cross.status() === 403 && (await cross.json()).error.code === "CSRF_ORIGIN_REJECTED");
    await api.dispose();
    const cop = await pageFetch(pa, `/api/v1/cases/${caseA}/copilot`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question: "why?" }) });
    check("copilot FEATURE_DISABLED", cop.status === 503 && cop.body.error.code === "FEATURE_DISABLED");

    // 5. visitor isolation, forged cookie, revoked session, upload limit
    const b = await browser.newContext();
    const pb = await b.newPage();
    const anon = await pb.goto(`${BASE}/api/v1/cases/${caseA}`);
    check("no session -> SESSION_REQUIRED", anon.status() === 401 && (await anon.json()).error.code === "SESSION_REQUIRED");
    await useSample(pb, "SIMULATED_coldroom_24h_1min.csv");
    const foreign = await pageFetch(pb, `/api/v1/cases/${caseA}/quality`);
    check("visitor B gets CASE_NOT_FOUND for A's case", foreign.status === 404 && foreign.body.error.code === "CASE_NOT_FOUND");
    const c = await browser.newContext();
    await c.addCookies([{ name: "cs_session", value: `${sess.value.slice(0, -4)}AAAA`, url: BASE }]);
    const forged = await (await c.newPage()).goto(`${BASE}/api/v1/cases/${caseA}`);
    check("forged cookie -> SESSION_INVALID", forged.status() === 401 && (await forged.json()).error.code === "SESSION_INVALID");
    const big = await pa.evaluate(async () => {
      const t = await (await fetch("/api/v1/csrf")).json();
      const r = await fetch("/api/v1/cases?filename=big.csv", { method: "POST", body: new Blob([new Uint8Array(10 * 1024 * 1024 + 1)]), headers: { "Content-Type": "text/csv", "X-CSRF-Token": t.csrf_token } });
      return { status: r.status, body: await r.json() };
    });
    check("oversized upload -> UPLOAD_TOO_LARGE", big.status === 413 && big.body.error.code === "UPLOAD_TOO_LARGE");
    const end = await pageFetch(pa, "/api/v1/session", { method: "DELETE" });
    const after = await pageFetch(pa, `/api/v1/cases/${caseA}`);
    check("ended session is revoked", end.status === 204 && after.status === 401, `${end.status} ${JSON.stringify(after.body)}`);
  } finally {
    await browser.close();
  }
  const failed = results.filter((r) => !r.ok);
  console.log(`\n${results.length - failed.length}/${results.length} checks passed`);
  process.exit(failed.length ? 1 : 0);
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
