const state = {
  sessionId: null,
  csrfToken: null,
  copilotEnabled: true,
  activeCaseId: null,
  activeCaseSummary: null,
  unlockedStep: 1,
  dataQuality: null,
  timelinePoints: [],
  events: [],
  prediction: null,
  evidenceBoard: null,
  hypotheses: [],
  nextBestChecks: [],
  stagedDraft: null,
  stagedToken: null,
  confirmedFindings: [],
  incident: null,
  report: null,
};

function showToast(message, durationMs = 3600) {
  const banner = document.getElementById("toast-banner");
  if (!banner) return;
  banner.textContent = message;
  banner.classList.add("visible");
  clearTimeout(showToast._timer);
  showToast._timer = setTimeout(() => {
    banner.classList.remove("visible");
  }, durationMs);
}

async function apiRequest(method, path, body = null) {
  const headers = { "Content-Type": "application/json" };
  if (state.csrfToken && ["POST", "PUT", "PATCH", "DELETE"].includes(method.toUpperCase())) {
    headers["X-CSRF-Token"] = state.csrfToken;
  }
  const opts = {
    method,
    headers,
    credentials: "same-origin",
  };
  if (body !== null) {
    opts.body = JSON.stringify(body);
  }
  const resp = await fetch(path, opts);
  const data = await resp.json();
  return { status: resp.status, data };
}

function updateNavigationLocks(unlockedStep) {
  state.unlockedStep = unlockedStep || 1;
  const tabs = document.querySelectorAll("#workflow-anchor-nav .anchor-tab");
  tabs.forEach((btn) => {
    const reqStep = parseInt(btn.getAttribute("data-step") || "1", 10);
    if (reqStep <= state.unlockedStep) {
      btn.classList.remove("locked");
      btn.classList.add("unlocked");
    } else {
      btn.classList.add("locked");
      btn.classList.remove("unlocked");
    }
  });
}

function switchView(viewName) {
  const targetBtn = document.querySelector(`#workflow-anchor-nav .anchor-tab[data-view="${viewName}"]`);
  if (targetBtn && targetBtn.classList.contains("locked")) {
    const reqStep = targetBtn.getAttribute("data-step");
    showToast(`Step locked: complete upstream workflow steps first (requires Step ${reqStep} unlocked).`);
    return;
  }

  document.querySelectorAll(".view-section").forEach((sec) => sec.classList.remove("active"));
  const targetSec = document.getElementById(`view-${viewName}`);
  if (targetSec) {
    targetSec.classList.add("active");
  }

  document.querySelectorAll("#workflow-anchor-nav .anchor-tab").forEach((btn) => {
    btn.classList.toggle("active", btn.getAttribute("data-view") === viewName);
  });

  if (viewName === "incident" && state.activeCaseId && state.unlockedStep >= 6) {
    loadIncidentView();
  } else if (viewName === "reports" && state.activeCaseId && state.unlockedStep >= 10) {
    runStep10Report(false);
  }
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function updateHeaderBadge() {
  const provEl = document.getElementById("active-case-provenance-label");
  const titleEl = document.getElementById("active-case-title-label");
  if (!state.activeCaseSummary) {
    provEl.textContent = "NO ACTIVE CASE";
    titleEl.textContent = "Select or upload a log";
    return;
  }
  const c = state.activeCaseSummary;
  provEl.textContent = `${c.evidence_class}${c.user_claimed_real ? " (USER CLAIMED REAL · KEPT UNVALIDATED)" : ""}`;
  titleEl.textContent = `${c.title} (${c.case_id}) · Step ${c.unlocked_step}/10`;
}

async function initSessionAndDashboard() {
  const { status, data } = await apiRequest("GET", "/api/session");
  if (status === 200) {
    state.sessionId = data.session_id;
    state.csrfToken = data.csrf_token;
    state.copilotEnabled = data.copilot_enabled;
    updateCopilotGateUI();
  }
  await refreshDashboard();
}

async function refreshDashboard() {
  const { status, data } = await apiRequest("GET", "/api/dashboard");
  if (status !== 200) return;
  state.copilotEnabled = data.copilot_feature_enabled;
  updateCopilotGateUI();

  const tbody = document.getElementById("dashboard-cases-tbody");
  const cases = data.active_cases || [];
  if (cases.length === 0) {
    tbody.innerHTML = `
      <tr>
        <td colspan="8" style="color:var(--color-slate);">
          No cases in this session yet. Click <strong>Run Full 10-Step Pipeline</strong> or <strong>+ New Simulated Outage Case</strong> to begin.
        </td>
      </tr>
    `;
    return;
  }

  tbody.innerHTML = cases
    .map(
      (c) => `
      <tr>
        <td><span class="mono-code">${c.case_id}</span></td>
        <td><strong>${escapeHtml(c.title)}</strong></td>
        <td><span class="launch-status-label">${c.evidence_class}</span></td>
        <td>${c.user_claimed_real ? "Yes (Stored Separately)" : "No"}</td>
        <td>${c.point_count} pts</td>
        <td>Step ${c.unlocked_step} / 10</td>
        <td>${c.confirmed_findings_count} confirmed</td>
        <td>
          <button type="button" class="pill-outlined" onclick="selectExistingCase('${c.case_id}')">
            Open Case
          </button>
        </td>
      </tr>
    `
    )
    .join("");
}

async function selectExistingCase(caseId) {
  const { status, data } = await apiRequest("GET", "/api/dashboard");
  if (status !== 200) return;
  const found = (data.active_cases || []).find((c) => c.case_id === caseId);
  if (!found) return;
  state.activeCaseId = found.case_id;
  state.activeCaseSummary = found;
  updateNavigationLocks(found.unlocked_step);
  updateHeaderBadge();
  await runStep2DataQuality(false);
  showToast(`Loaded case ${found.case_id} (${found.evidence_class}).`);
}

async function launchDemoScenario(scenario = "power_cut") {
  const { status, data } = await apiRequest("POST", "/api/cases/demo", { scenario });
  if (status !== 201) {
    showToast(`Failed to create demo case: ${data.message || status}`);
    return null;
  }
  state.activeCaseId = data.case_id;
  state.activeCaseSummary = data;
  state.confirmedFindings = [];
  state.stagedDraft = null;
  state.stagedToken = null;
  updateNavigationLocks(data.unlocked_step);
  updateHeaderBadge();
  await refreshDashboard();
  await runStep2DataQuality(true);
  return data;
}

function handleFileInputLoad(event) {
  const file = event.target.files && event.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (e) => {
    document.getElementById("upload-raw-content").value = e.target.result;
    if (file.name.toLowerCase().endsWith(".json")) {
      document.getElementById("upload-format").value = "json";
    } else {
      document.getElementById("upload-format").value = "csv";
    }
    document.getElementById("upload-title").value = file.name;
  };
  reader.readAsText(file);
}

async function handleUploadCase(event) {
  event.preventDefault();
  const title = document.getElementById("upload-title").value;
  const format = document.getElementById("upload-format").value;
  const critical_threshold_c = parseFloat(document.getElementById("upload-threshold").value || "8.0");
  const raw_content = document.getElementById("upload-raw-content").value;
  const user_claimed_real = document.getElementById("upload-claim-real").checked;

  const { status, data } = await apiRequest("POST", "/api/cases/upload", {
    title,
    format,
    critical_threshold_c,
    raw_content,
    user_claimed_real,
  });

  if (status !== 201) {
    showToast(`Upload error: ${data.message || "Invalid log"}`);
    return;
  }
  state.activeCaseId = data.case_id;
  state.activeCaseSummary = data;
  updateNavigationLocks(data.unlocked_step);
  updateHeaderBadge();
  await refreshDashboard();
  showToast(
    `Uploaded ${data.case_id}: tagged strictly as ${data.evidence_class} (user_claimed_real=${data.user_claimed_real}).`
  );
  await runStep2DataQuality(true);
}

async function runStep2DataQuality(navigate = true) {
  if (!state.activeCaseId) {
    showToast("Create or upload a case in Step 1 first.");
    return;
  }
  const { status, data } = await apiRequest("POST", `/api/cases/${state.activeCaseId}/data-quality`);
  if (status !== 200) {
    showToast(`Data Quality error: ${data.message}`);
    return;
  }
  const dq = data.data_quality;
  state.dataQuality = dq;
  state.activeCaseSummary.unlocked_step = data.unlocked_step;
  updateNavigationLocks(data.unlocked_step);
  updateHeaderBadge();

  const grid = document.getElementById("dq-metrics-grid");
  grid.innerHTML = `
    <div class="feature-card">
      <div class="launch-status-label">EVIDENCE CLASS · ${dq.evidence_class}</div>
      <div class="card-metric-48" style="margin-top:8px;">${dq.valid_rows} <span style="font-size:18px;color:var(--color-slate);">/ ${dq.total_rows}</span></div>
      <p style="font-size:14px; color:var(--color-slate); margin-top:6px;">Valid timestamped readings (${(dq.completeness_ratio * 100).toFixed(1)}% completeness)</p>
    </div>
    <div class="feature-card">
      <div class="launch-status-label">SAMPLING CADENCE</div>
      <div class="card-metric-48" style="margin-top:8px;">${dq.median_cadence_sec}s</div>
      <p style="font-size:14px; color:var(--color-slate); margin-top:6px;">Median interval · Jitter ±${dq.cadence_jitter_sec}s</p>
    </div>
    <div class="feature-card">
      <div class="launch-status-label">CONTINUITY AUDIT</div>
      <div class="card-metric-48" style="margin-top:8px;">${dq.gap_count}</div>
      <p style="font-size:14px; color:var(--color-slate); margin-top:6px;">Sampling gaps detected · Max gap ${dq.max_gap_sec}s</p>
    </div>
    <div class="feature-card">
      <div class="launch-status-label">SENSOR HEALTH</div>
      <div class="card-metric-48" style="margin-top:8px;">${dq.is_usable_for_analysis ? "PASS" : "FAIL"}</div>
      <p style="font-size:14px; color:var(--color-slate); margin-top:6px;">Out-of-range: ${dq.out_of_range_count} · Flatlines: ${dq.flatline_segments}</p>
    </div>
  `;

  const issuesEl = document.getElementById("dq-issues-container");
  if (!dq.issues || dq.issues.length === 0) {
    issuesEl.innerHTML = `
      <div class="feature-card-bordered" style="padding:18px;">
        <div class="launch-status-label">ZERO ANOMALIES</div>
        <p style="font-size:14px; margin-top:4px;">All ${dq.valid_rows} sensor readings passed physical range [-60 °C, +60 °C], cadence continuity, and flatline checks.</p>
      </div>
    `;
  } else {
    issuesEl.innerHTML = dq.issues
      .map(
        (iss) => `
        <div class="feature-card-bordered" style="padding:16px; margin-bottom:10px;">
          <div class="launch-status-label">${iss.severity} · ${iss.code}</div>
          <p style="font-size:14px; margin-top:4px;">${escapeHtml(iss.message)}</p>
        </div>
      `
      )
      .join("");
  }

  if (navigate) switchView("data-quality");
}

async function runStep3Events(navigate = true) {
  if (!state.activeCaseId) return;
  const { status, data } = await apiRequest("POST", `/api/cases/${state.activeCaseId}/events`);
  if (status !== 200) {
    showToast(`Step 3 locked or failed: ${data.message}`);
    return;
  }
  state.timelinePoints = data.points || [];
  state.events = data.events || [];
  state.activeCaseSummary.unlocked_step = data.unlocked_step;
  updateNavigationLocks(data.unlocked_step);
  updateHeaderBadge();

  document.getElementById("timeline-provenance-tag").textContent = `PROVENANCE: ${data.evidence_class}`;
  renderTimelineSvg(state.timelinePoints, state.events, state.activeCaseSummary.critical_threshold_c || 8.0);

  const tbody = document.getElementById("timeline-events-tbody");
  if (state.events.length === 0) {
    tbody.innerHTML = `<tr><td colspan="7" style="color:var(--color-slate);">No thermal excursion or drift events detected.</td></tr>`;
  } else {
    tbody.innerHTML = state.events
      .map(
        (ev) => `
        <tr>
          <td><span class="mono-code">${ev.event_id}</span></td>
          <td><strong>${ev.event_type}</strong></td>
          <td><span class="launch-status-label">${ev.evidence_class}</span></td>
          <td>${(ev.duration_sec / 60).toFixed(1)} min</td>
          <td>${ev.start_temp_c.toFixed(2)} °C → <strong>${ev.peak_temp_c.toFixed(2)} °C</strong></td>
          <td>${ev.drift_rate_c_per_hour >= 0 ? "+" : ""}${ev.drift_rate_c_per_hour.toFixed(2)} °C/h</td>
          <td>
            <button type="button" class="pill-outlined" onclick="runStep4Prediction('${ev.event_id}', true)">
              Forecast Event
            </button>
          </td>
        </tr>
      `
      )
      .join("");
  }

  if (navigate) switchView("timeline");
}

function renderTimelineSvg(points, events, thresholdC) {
  const container = document.getElementById("timeline-svg-container");
  if (!points || points.length < 2) {
    container.innerHTML = `<p style="color:var(--color-slate);">Insufficient points to render timeline.</p>`;
    return;
  }
  const W = 1040, H = 300, padL = 54, padR = 24, padT = 24, padB = 38;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  const tMin = points[0].timestamp_sec;
  const tMax = Math.max(tMin + 1, points[points.length - 1].timestamp_sec);
  const temps = points.map((p) => p.temp_c);
  const yMin = Math.min(-2, Math.floor(Math.min(...temps) - 1));
  const yMax = Math.max(thresholdC + 4, Math.ceil(Math.max(...temps) + 2));

  const xScale = (t) => padL + ((t - tMin) / (tMax - tMin)) * plotW;
  const yScale = (y) => padT + plotH - ((y - yMin) / Math.max(1e-6, yMax - yMin)) * plotH;

  const eventRects = (events || [])
    .map((ev) => {
      const x1 = xScale(ev.start_timestamp_sec);
      const x2 = Math.max(x1 + 4, xScale(ev.end_timestamp_sec));
      const fill =
        ev.event_type === "POWER_CUT"
          ? "rgba(182, 68, 0, 0.11)"
          : ev.event_type === "COOLING_RECOVERY"
          ? "rgba(0, 113, 227, 0.09)"
          : "rgba(134, 134, 139, 0.12)";
      return `<rect x="${x1.toFixed(1)}" y="${padT}" width="${(x2 - x1).toFixed(1)}" height="${plotH}" fill="${fill}" />`;
    })
    .join("");

  const pathD = points
    .map((p, idx) => `${idx === 0 ? "M" : "L"} ${xScale(p.timestamp_sec).toFixed(1)} ${yScale(p.temp_c).toFixed(1)}`)
    .join(" ");

  const threshY = yScale(thresholdC);

  container.innerHTML = `
    <svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="Cold-Room Temperature Timeline">
      ${eventRects}
      <line x1="${padL}" y1="${padT + plotH}" x2="${W - padR}" y2="${padT + plotH}" stroke="#d6d6d6" stroke-width="1"/>
      <line x1="${padL}" y1="${padT}" x2="${padL}" y2="${padT + plotH}" stroke="#d6d6d6" stroke-width="1"/>
      <line x1="${padL}" y1="${threshY.toFixed(1)}" x2="${W - padR}" y2="${threshY.toFixed(1)}" stroke="#b64400" stroke-width="1.5" stroke-dasharray="5 5"/>
      <text x="${W - padR - 6}" y="${(threshY - 6).toFixed(1)}" text-anchor="end" font-size="11" font-weight="600" fill="#b64400">Critical Limit ${thresholdC.toFixed(1)} °C</text>
      <path d="${pathD}" fill="none" stroke="#1d1d1f" stroke-width="2.2" />
      <text x="${padL - 8}" y="${padT + 10}" text-anchor="end" font-size="11" fill="#707070">${yMax}°C</text>
      <text x="${padL - 8}" y="${padT + plotH}" text-anchor="end" font-size="11" fill="#707070">${yMin}°C</text>
      <text x="${padL}" y="${H - 10}" font-size="11" fill="#707070">${points[0].iso_timestamp.slice(11, 16)} UTC</text>
      <text x="${W - padR}" y="${H - 10}" text-anchor="end" font-size="11" fill="#707070">${points[points.length - 1].iso_timestamp.slice(11, 16)} UTC (${points.length} samples)</text>
    </svg>
  `;
}

async function runStep4Prediction(eventId = null, navigate = true) {
  if (!state.activeCaseId) return;
  const payload = { horizon_sec: 3600.0 };
  if (eventId) payload.event_id = eventId;

  const { status, data } = await apiRequest("POST", `/api/cases/${state.activeCaseId}/prediction`, payload);
  if (status !== 200) {
    showToast(`Step 4 error: ${data.message}`);
    return;
  }
  const pred = data.prediction;
  state.prediction = pred;
  state.activeCaseSummary.unlocked_step = data.unlocked_step;
  updateNavigationLocks(data.unlocked_step);
  updateHeaderBadge();

  const container = document.getElementById("prediction-outcome-container");
  if (pred.response_type === "REFUSAL") {
    const failedList = (pred.failed_checks || []).map((c) => `<li>${escapeHtml(c)}</li>`).join("");
    container.innerHTML = `
      <div class="feature-card" style="border: 1px solid var(--color-launch-orange);">
        <div class="launch-status-label">RELIABILITY GATE REFUSAL · ${pred.reason_code} · PROVENANCE: ${pred.evidence_class}</div>
        <h3 class="feature-heading" style="margin-top:8px; margin-bottom:12px;">Forecast Refused. Zero Guessed Curves.</h3>
        <p style="font-size:17px; color:var(--color-ink); margin-bottom:20px;">
          ${escapeHtml(pred.refusal_explanation)}
        </p>
        <div class="feature-card-bordered" style="margin-bottom:20px;">
          <div class="launch-status-label">FAILED RELIABILITY CHECKS</div>
          <ul style="margin-top:8px; padding-left:20px; font-size:14px; color:var(--color-ink);">
            ${failedList}
          </ul>
        </div>
        <p style="font-size:13px; color:var(--color-slate);">
          Rule 2 Guardrail: Success and Refusal are strictly separate API response types. When the Reliability Gate refuses a case, ColdSense never outputs dummy coordinates or fallback curves.
        </p>
      </div>
    `;
  } else {
    const timeToBreachMin = pred.estimated_time_to_threshold_sec
      ? `${(pred.estimated_time_to_threshold_sec / 60).toFixed(1)} min`
      : "Not breached";
    container.innerHTML = `
      <div class="grid-4" style="margin-bottom: var(--spacing-28);">
        <div class="feature-card">
          <div class="launch-status-label">THERMAL TIME CONSTANT (τ)</div>
          <div class="card-metric-48" style="margin-top:8px;">${pred.time_constant_tau_hours} h</div>
          <p style="font-size:14px; color:var(--color-slate); margin-top:6px;">${pred.time_constant_tau_sec}s toward ${pred.ambient_temp_c} °C ambient</p>
        </div>
        <div class="feature-card">
          <div class="launch-status-label">GOODNESS OF FIT (R²)</div>
          <div class="card-metric-48" style="margin-top:8px;">${pred.r_squared}</div>
          <p style="font-size:14px; color:var(--color-slate); margin-top:6px;">95% Uncertainty Half-Width: ±${pred.uncertainty_95_halfwidth_c} °C</p>
        </div>
        <div class="feature-card">
          <div class="launch-status-label">MODEL VS. NAIVE BASELINE RMSE</div>
          <div class="card-metric-48" style="margin-top:8px;">${pred.thermal_model_rmse_c}°C</div>
          <p style="font-size:14px; color:var(--color-slate); margin-top:6px;">Naive linear baseline RMSE: ${pred.naive_baseline_rmse_c} °C</p>
        </div>
        <div class="feature-card">
          <div class="launch-status-label">TIME TO ${pred.critical_threshold_c} °C BREACH</div>
          <div class="card-metric-48" style="margin-top:8px;">${timeToBreachMin}</div>
          <p style="font-size:14px; color:var(--color-slate); margin-top:6px;">Case Class: ${pred.evidence_class} · Curve: ${pred.baseline_evidence_class}</p>
        </div>
      </div>

      <div class="chart-stage">
        <div style="display:flex; justify-content:space-between; align-items:baseline; margin-bottom:12px;">
          <h3 class="card-title-24" style="margin-bottom:0;">First-Order Thermal Drift Forecast with 50% &amp; 95% Uncertainty Bands</h3>
          <span class="launch-status-label">RELIABILITY GATE PASSED · EVENT ${pred.event_id}</span>
        </div>
        <div id="prediction-fan-svg-container"></div>
        <div class="chart-legend">
          <span class="legend-item"><span style="width:12px;height:3px;background:#0066cc;display:inline-block;"></span> First-Order Thermal Model (EXPECTED)</span>
          <span class="legend-item"><span style="width:12px;height:10px;background:rgba(0,113,227,0.14);display:inline-block;"></span> 95% Confidence Band</span>
          <span class="legend-item"><span style="width:12px;height:3px;background:#86868b;display:inline-block;"></span> Naive Linear Baseline</span>
          <span class="legend-item"><span style="width:8px;height:8px;border-radius:50%;background:#1d1d1f;display:inline-block;"></span> Observed Telemetry (${pred.evidence_class})</span>
        </div>
      </div>
    `;
    renderPredictionFanSvg(pred);
  }

  if (navigate) switchView("prediction");
}

function renderPredictionFanSvg(pred) {
  const container = document.getElementById("prediction-fan-svg-container");
  const curve = pred.forecast_curve || [];
  if (!curve.length) return;

  const W = 1040, H = 320, padL = 54, padR = 24, padT = 24, padB = 38;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  const maxElapsed = Math.max(1, curve[curve.length - 1].elapsed_sec);
  const allVals = [];
  curve.forEach((pt) => {
    allVals.push(pt.lower_95_c, pt.upper_95_c, pt.naive_baseline_temp_c);
    if (pt.observed_temp_c !== null) allVals.push(pt.observed_temp_c);
  });
  const yMin = Math.floor(Math.min(...allVals, 0));
  const yMax = Math.ceil(Math.max(...allVals, pred.critical_threshold_c + 2));

  const xScale = (el) => padL + (el / maxElapsed) * plotW;
  const yScale = (y) => padT + plotH - ((y - yMin) / Math.max(1e-6, yMax - yMin)) * plotH;

  const upperPts = curve.map((p) => `${xScale(p.elapsed_sec).toFixed(1)},${yScale(p.upper_95_c).toFixed(1)}`);
  const lowerPts = [...curve].reverse().map((p) => `${xScale(p.elapsed_sec).toFixed(1)},${yScale(p.lower_95_c).toFixed(1)}`);
  const poly95 = [...upperPts, ...lowerPts].join(" ");

  const modelPath = curve
    .map((p, i) => `${i === 0 ? "M" : "L"} ${xScale(p.elapsed_sec).toFixed(1)} ${yScale(p.thermal_model_temp_c).toFixed(1)}`)
    .join(" ");
  const naivePath = curve
    .map((p, i) => `${i === 0 ? "M" : "L"} ${xScale(p.elapsed_sec).toFixed(1)} ${yScale(p.naive_baseline_temp_c).toFixed(1)}`)
    .join(" ");

  const obsDots = curve
    .filter((p) => p.observed_temp_c !== null)
    .filter((_, idx) => idx % 2 === 0)
    .map(
      (p) =>
        `<circle cx="${xScale(p.elapsed_sec).toFixed(1)}" cy="${yScale(p.observed_temp_c).toFixed(1)}" r="3.2" fill="#1d1d1f" />`
    )
    .join("");

  const threshY = yScale(pred.critical_threshold_c);

  container.innerHTML = `
    <svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="Thermal Drift Prediction Curve">
      <polygon points="${poly95}" fill="rgba(0, 113, 227, 0.14)" />
      <line x1="${padL}" y1="${padT + plotH}" x2="${W - padR}" y2="${padT + plotH}" stroke="#d6d6d6" stroke-width="1"/>
      <line x1="${padL}" y1="${padT}" x2="${padL}" y2="${padT + plotH}" stroke="#d6d6d6" stroke-width="1"/>
      <line x1="${padL}" y1="${threshY.toFixed(1)}" x2="${W - padR}" y2="${threshY.toFixed(1)}" stroke="#b64400" stroke-width="1.5" stroke-dasharray="4 4"/>
      <text x="${W - padR - 6}" y="${(threshY - 6).toFixed(1)}" text-anchor="end" font-size="11" font-weight="600" fill="#b64400">Critical Threshold ${pred.critical_threshold_c} °C</text>
      <path d="${naivePath}" fill="none" stroke="#86868b" stroke-width="1.8" stroke-dasharray="6 5" />
      <path d="${modelPath}" fill="none" stroke="#0066cc" stroke-width="2.8" />
      ${obsDots}
      <text x="${padL - 8}" y="${padT + 10}" text-anchor="end" font-size="11" fill="#707070">${yMax}°C</text>
      <text x="${padL - 8}" y="${padT + plotH}" text-anchor="end" font-size="11" fill="#707070">${yMin}°C</text>
      <text x="${padL}" y="${H - 10}" font-size="11" fill="#707070">Elapsed 0 min</text>
      <text x="${W - padR}" y="${H - 10}" text-anchor="end" font-size="11" fill="#707070">Horizon +${(maxElapsed / 60).toFixed(0)} min</text>
    </svg>
  `;
}

async function runStep5Evidence(navigate = true) {
  if (!state.activeCaseId) return;
  const { status, data } = await apiRequest("GET", `/api/cases/${state.activeCaseId}/evidence`);
  if (status !== 200) {
    showToast(`Step 5 error: ${data.message}`);
    return;
  }
  const board = data.evidence_board;
  state.evidenceBoard = board;
  state.activeCaseSummary.unlocked_step = data.unlocked_step;
  updateNavigationLocks(data.unlocked_step);
  updateHeaderBadge();

  const renderCol = (heading, kicker, items) => `
    <div class="feature-card">
      <div class="launch-status-label">${kicker}</div>
      <h3 class="card-title-24" style="margin-top:6px; margin-bottom:16px;">${heading}</h3>
      ${(items || [])
        .map(
          (item) => `
          <div class="feature-card-bordered" style="padding:18px; margin-bottom:14px;">
            <div style="display:flex; justify-content:space-between; align-items:baseline;">
              <span class="launch-status-label">CLASS: ${item.evidence_class}</span>
              <span class="mono-code">${item.evidence_id}</span>
            </div>
            <h4 style="font-size:16px; font-weight:600; margin:6px 0;">${escapeHtml(item.title)}</h4>
            <p style="font-size:14px; color:var(--color-ink); margin-bottom:10px;">${escapeHtml(item.statement)}</p>
            <div style="font-size:12px; color:var(--color-slate);">
              <div><strong>Provenance:</strong> <span class="mono-code">${escapeHtml(item.provenance_source)}</span></div>
              <div style="margin-top:4px;"><strong>Supports:</strong> ${escapeHtml((item.supports || []).join("; "))}</div>
              <div style="margin-top:2px;"><strong>Does NOT support:</strong> ${escapeHtml((item.does_not_support || []).join("; "))}</div>
            </div>
          </div>
        `
        )
        .join("")}
    </div>
  `;

  document.getElementById("evidence-columns-container").innerHTML =
    renderCol("1. Observed Facts", "SENSOR TELEMETRY FACTS", board.facts) +
    renderCol("2. Model Inferences", "PHYSICS MODEL PARAMETERS", board.model_inferences) +
    renderCol("3. Recommendations", "ACTIONABLE GUIDANCE", board.recommendations);

  if (navigate) switchView("evidence");
}

async function runStep6NextBestCheck(navigate = true) {
  if (!state.activeCaseId) return;
  const { status, data } = await apiRequest("GET", `/api/cases/${state.activeCaseId}/next-best-check`);
  if (status !== 200) {
    showToast(`Step 6 error: ${data.message}`);
    return;
  }
  state.hypotheses = data.hypotheses || [];
  state.nextBestChecks = data.next_best_checks || [];
  state.activeCaseSummary.unlocked_step = data.unlocked_step;
  updateNavigationLocks(data.unlocked_step);
  updateHeaderBadge();
  populateStageFindingSelectors();

  const hypsHtml = state.hypotheses
    .map(
      (h) => `
      <div class="feature-card-bordered" style="padding:20px; margin-bottom:14px;">
        <div style="display:flex; justify-content:space-between; align-items:baseline;">
          <span class="launch-status-label">STATUS: ${h.verification_status} · ${h.evidence_class}</span>
          <span class="mono-code">${h.template_id}</span>
        </div>
        <h4 style="font-size:17px; font-weight:600; margin:6px 0;">${escapeHtml(h.title)}</h4>
        <p style="font-size:14px; color:var(--color-ink); margin-bottom:10px;">${escapeHtml(h.rendered_statement)}</p>
        <div style="font-size:12px; color:var(--color-slate);">
          Rule 7 Guarantee: Rendered deterministically from template <span class="mono-code">${h.template_id}</span> (never written by an LLM).
        </div>
      </div>
    `
    )
    .join("");

  const checksHtml = state.nextBestChecks
    .map(
      (chk) => `
      <div class="feature-card-bordered" style="padding:20px; margin-bottom:14px;">
        <div style="display:flex; justify-content:space-between; align-items:baseline;">
          <span class="launch-status-label">PRIORITY RANK #${chk.rank} · EST. ${chk.estimated_minutes} MIN</span>
          <span class="mono-code">${chk.check_id}</span>
        </div>
        <h4 style="font-size:17px; font-weight:600; margin:6px 0;">${escapeHtml(chk.title)}</h4>
        <p style="font-size:14px; color:var(--color-ink); margin-bottom:10px;">${escapeHtml(chk.inspection_procedure)}</p>
        <div style="font-size:12px; color:var(--color-slate); margin-bottom:12px;">
          <div><strong>Confirm Indicator:</strong> ${escapeHtml(chk.confirm_indicator)}</div>
          <div style="margin-top:2px;"><strong>Reject Indicator:</strong> ${escapeHtml(chk.reject_indicator)}</div>
        </div>
        <button type="button" class="pill-pricing-blue" onclick="jumpToStageFinding('${chk.target_hypothesis_ids[0] || ""}', '${chk.check_id}')">
          Stage Technician Finding for Check #${chk.rank}
        </button>
      </div>
    `
    )
    .join("");

  document.getElementById("nbc-container").innerHTML = `
    <div class="feature-card">
      <div class="launch-status-label">TEMPLATE-BOUND CANDIDATES</div>
      <h3 class="card-title-24" style="margin-top:6px; margin-bottom:16px;">Open Diagnostic Hypotheses</h3>
      ${hypsHtml}
    </div>
    <div class="feature-card">
      <div class="launch-status-label">ORDERED BY INFORMATION GAIN</div>
      <h3 class="card-title-24" style="margin-top:6px; margin-bottom:16px;">Physical Inspection Sequence</h3>
      ${checksHtml}
    </div>
  `;

  if (navigate) switchView("next-best-check");
}

function updateCopilotGateUI() {
  const badge = document.getElementById("copilot-gate-badge");
  if (!badge) return;
  badge.textContent = state.copilotEnabled
    ? "SERVER GATE: ENABLED (LANGGRAPH + GEMINI)"
    : "SERVER GATE: FEATURE_DISABLED (503) - set COLDSENSE_COPILOT_ENABLED on the server";
}

function engineBannerHtml(data) {
  // Which engine produced this answer, always visible; plus a notice when Gemini did not answer.
  const engine = data.engine || "UNKNOWN";
  const notice = data.engine_notice;
  const warn = engine !== "GEMINI" && notice;
  return `
    <div class="launch-status-label" style="margin-bottom:8px;">ENGINE: <span class="mono-code">${escapeHtml(engine)}</span></div>
    ${
      warn
        ? `<div class="feature-card-bordered" style="padding:10px 14px; margin-bottom:12px; border-color:var(--color-launch-orange);">
             <p style="font-size:13px; font-weight:600; color:var(--color-launch-orange); margin:0;">${escapeHtml(notice)}</p>
           </div>`
        : ""
    }
  `;
}

function submitCopilotQuery(event) {
  return handleCopilotSubmit(event);
}

function checkKbStatus() {
  return runDirectKbAudit();
}

const COPILOT_PRESETS = {
  gasket: "What is the door gasket gap tolerance and paper strip check?",
  threshold: "What is the critical temperature threshold and excursion time limit?",
  hazard_en: "How to bypass the high pressure switch?",
  hazard_fr: "Comment bypasser le pressostat de sécurité haute pression?",
  conflict: "What is the defrost interval of the EX-200 units built from 2024?",
  unanswerable: "What is the serial number of the warehouse forklift?",
};

function fillCopilotPreset(queryText) {
  document.getElementById("copilot-query-input").value = queryText;
}

function askCopilotPreset(presetKey) {
  const queryText = COPILOT_PRESETS[presetKey] || presetKey;
  document.getElementById("copilot-query-input").value = queryText;
  handleCopilotSubmit(new Event("submit"));
}

async function handleGlobalKbSearch(event) {
  event.preventDefault();
  const q = document.getElementById("global-kb-search-input").value.trim();
  if (!q) return;
  document.getElementById("copilot-query-input").value = q;
  switchView("copilot");
  await runDirectKbAudit();
}

async function handleCopilotSubmit(event) {
  if (event && event.preventDefault) event.preventDefault();
  const query = document.getElementById("copilot-query-input").value.trim();
  if (!query) return;

  const { status, data } = await apiRequest("POST", "/api/copilot/ask", {
    query,
    case_id: state.activeCaseId,
  });

  const panel = document.getElementById("copilot-response-panel");
  const res = data.result;
  if (!res) return;

  if (res.response_type === "HAZARD_TEMPLATE" || res.response_type === "REFUSAL") {
    panel.innerHTML = `
      ${engineBannerHtml(data)}
      <div class="launch-status-label">${res.response_type} · ${res.reason_code} · HTTP ${status}</div>
      <h3 class="card-title-24" style="margin-top:6px;">Guarded Refusal / Safety Intercept</h3>
      <div class="feature-card-bordered" style="padding:18px; margin-top:12px; border-color:var(--color-launch-orange);">
        <p style="font-size:15px; font-weight:500; color:var(--color-ink);">${escapeHtml(res.message)}</p>
        ${
          res.hazard_matched_terms && res.hazard_matched_terms.length
            ? `<p style="font-size:12px; color:var(--color-launch-orange); margin-top:8px;">Matched Hazard Gate Terms: <span class="mono-code">${escapeHtml(res.hazard_matched_terms.join(", "))}</span> (Zero LLM or tool calls executed)</p>`
            : ""
        }
      </div>
    `;
    return;
  }

  const citsHtml = (res.verbatim_citations || [])
    .map(
      (c) => `
      <div class="verbatim-quote-block">
        <div style="font-size:12px; font-weight:600; color:var(--color-slate);">
          [${escapeHtml(c.doc_id)}] ${escapeHtml(c.doc_title)} — ${escapeHtml(c.section)} (RRF Score: ${c.relevance_score})
        </div>
        <div style="margin-top:4px;">"${escapeHtml(c.verbatim_quote)}"</div>
      </div>
    `
    )
    .join("");

  panel.innerHTML = `
    ${engineBannerHtml(data)}
    <div class="launch-status-label">GROUNDED ANSWER · MODE: ${res.retrieval_mode} · CLASS: ${res.evidence_class}</div>
    <h3 class="card-title-24" style="margin-top:6px;">Verified Response</h3>
    <p style="font-size:15px; color:var(--color-ink); margin:12px 0 18px;">${escapeHtml(res.answer_text)}</p>
    <h4 style="font-size:13px; font-weight:600; color:var(--color-slate); text-transform:uppercase;">Verbatim Source Citations</h4>
    ${citsHtml}
    <div style="margin-top:14px; font-size:12px; color:var(--color-slate);">
      <div><strong>Numeric Integrity Audit Passed:</strong> Allowed numbers used = <span class="mono-code">[${(res.allowed_numbers_used || []).join(", ")}]</span></div>
      <div style="margin-top:4px;"><strong>Conditions:</strong> ${escapeHtml((res.conditions || []).join(" "))}</div>
    </div>
  `;
}

async function runDirectKbAudit() {
  const query = document.getElementById("copilot-query-input").value.trim();
  if (!query) return;
  const { status, data } = await apiRequest("POST", "/api/kb/query", { query });
  if (status !== 200) return;

  const panel = document.getElementById("copilot-response-panel");
  if (data.status === "CONFLICTING_SOURCES") {
    const confHtml = (data.conflicts || [])
      .map((cf) =>
        (cf.values || [])
          .map(
            (v) => `
          <div class="verbatim-quote-block">
            <div style="font-size:12px; font-weight:600; color:var(--color-launch-orange);">
              CONFLICTING REVISION · ${escapeHtml(v.doc_id)} (Rev ${escapeHtml(v.revision)}) · Stated Value: ${escapeHtml(v.value)}
            </div>
            <div>"${escapeHtml(v.verbatim_quote)}"</div>
          </div>
        `
          )
          .join("")
      )
      .join("");
    panel.innerHTML = `
      <div class="launch-status-label">PHASE 7 KB STATUS · CONFLICTING_SOURCES</div>
      <h3 class="card-title-24" style="margin-top:6px;">Cross-Document Conflict Detected</h3>
      <p style="font-size:14px; color:var(--color-slate); margin-bottom:12px;">
        Two active documents for the same equipment model state different values in the same unit. Both revisions are shown without guessing one:
      </p>
      ${confHtml}
    `;
  } else if (data.status === "NO_RELEVANT_DOCUMENTATION") {
    panel.innerHTML = `
      <div class="launch-status-label">PHASE 7 KB STATUS · NO_RELEVANT_DOCUMENTATION</div>
      <h3 class="card-title-24" style="margin-top:6px;">Refused by Production Answerability Gate</h3>
      <p style="font-size:14px; color:var(--color-slate); margin-top:8px;">
        No active, unquarantined manual section passed the equipment model filter, exact-identifier gate, and quantity/action answerability verifier.
      </p>
    `;
  } else {
    const cits = (data.citations || [])
      .map(
        (c) => `
        <div class="verbatim-quote-block">
          <div style="font-size:12px; font-weight:600; color:var(--color-slate);">[${escapeHtml(c.doc_id)}] ${escapeHtml(c.doc_title)} (${escapeHtml(c.section)})</div>
          <div>"${escapeHtml(c.verbatim_quote)}"</div>
        </div>
      `
      )
      .join("");
    panel.innerHTML = `
      <div class="launch-status-label">PHASE 7 KB STATUS · ${data.status}</div>
      <h3 class="card-title-24" style="margin-top:6px;">Verified Extractive Passages</h3>
      ${cits}
    `;
  }
}

function populateStageFindingSelectors() {
  const hypSel = document.getElementById("stage-hypothesis-select");
  const chkSel = document.getElementById("stage-check-select");
  if (!hypSel || !chkSel) return;

  hypSel.innerHTML = (state.hypotheses || [])
    .map((h) => `<option value="${h.hypothesis_id}">${h.hypothesis_id} — ${escapeHtml(h.title)} [${h.verification_status}]</option>`)
    .join("");
  chkSel.innerHTML = (state.nextBestChecks || [])
    .map((c) => `<option value="${c.check_id}">${c.check_id} (Rank #${c.rank}) — ${escapeHtml(c.title)}</option>`)
    .join("");
}

function jumpToStageFinding(hypId, chkId) {
  populateStageFindingSelectors();
  if (hypId) document.getElementById("stage-hypothesis-select").value = hypId;
  if (chkId) document.getElementById("stage-check-select").value = chkId;
  switchView("investigation");
}

async function handleStageFinding(event) {
  event.preventDefault();
  if (!state.activeCaseId) return;
  const hypothesis_id = document.getElementById("stage-hypothesis-select").value;
  const check_id = document.getElementById("stage-check-select").value;
  const proposed_status = document.getElementById("stage-verdict-select").value;
  const technician_id = document.getElementById("stage-tech-id").value;
  const technician_observation = document.getElementById("stage-observation").value;

  const { status, data } = await apiRequest("POST", `/api/cases/${state.activeCaseId}/findings/stage`, {
    hypothesis_id,
    check_id,
    proposed_status,
    technician_id,
    technician_observation,
  });

  if (status !== 200) {
    showToast(`Blocked: ${data.message}`);
    return;
  }

  state.stagedDraft = data.draft_record;
  state.stagedToken = data.trusted_confirmation_token;

  document.getElementById("staged-token-details").innerHTML = `
    <div class="feature-card-bordered" style="padding:18px;">
      <div class="launch-status-label">SINGLE-USE TOKEN ISSUED · BOUND TO SHA-256 RECORD HASH</div>
      <div style="font-size:13px; margin-top:10px; display:flex; flex-direction:column; gap:6px;">
        <div><strong>Token ID:</strong> <span class="mono-code">${state.stagedToken.token_id}</span></div>
        <div><strong>SHA-256 Record Hash:</strong> <span class="mono-code">${state.stagedDraft.record_hash.slice(0, 32)}...</span></div>
        <div><strong>HMAC Signature:</strong> <span class="mono-code">${state.stagedToken.signature.slice(0, 32)}...</span></div>
        <div><strong>Proposed Verdict:</strong> <span class="launch-status-label">${state.stagedDraft.proposed_status}</span> by <span class="mono-code">${escapeHtml(state.stagedDraft.technician_id)}</span></div>
      </div>
      <div style="display:flex; gap:10px; margin-top:16px; flex-wrap:wrap;">
        <button type="button" class="pill-pricing-blue" onclick="executeConfirmFinding()">
          Confirm Finding with Single-Use Token
        </button>
        <button type="button" class="pill-outlined" onclick="executeConfirmFinding(true)">
          Test Replay / Tamper Guard
        </button>
      </div>
    </div>
  `;
  showToast(`Minted single-use token ${state.stagedToken.token_id}. Ready for Step 9 confirmation.`);
}

async function executeConfirmFinding(simulateReplayOrTamper = false) {
  if (!state.activeCaseId || !state.stagedDraft || !state.stagedToken) return;
  const d = state.stagedDraft;
  const t = state.stagedToken;

  const obs = simulateReplayOrTamper ? d.technician_observation + " [TAMPERED]" : d.technician_observation;
  const { status, data } = await apiRequest("POST", `/api/cases/${state.activeCaseId}/findings/confirm`, {
    hypothesis_id: d.hypothesis_id,
    check_id: d.check_id,
    proposed_status: d.proposed_status,
    technician_id: d.technician_id,
    technician_observation: obs,
    token_id: t.token_id,
    signature: t.signature,
  });

  if (status !== 200) {
    showToast(`Cryptographic Guard Blocked Request: ${data.message}`);
    return;
  }

  const f = data.confirmed_finding;
  state.confirmedFindings.push(f);
  await runStep6NextBestCheck(false);
  await refreshDashboard();
  renderConfirmedFindingsList();
  showToast(`Confirmed finding ${f.finding_id} committed to append-only SQLite store.`);
}

function renderConfirmedFindingsList() {
  const el = document.getElementById("confirmed-findings-list");
  if (!state.confirmedFindings.length) {
    el.innerHTML = "No findings confirmed yet. All hypotheses remain UNKNOWN.";
    return;
  }
  el.innerHTML = state.confirmedFindings
    .map(
      (f) => `
      <div class="feature-card-bordered" style="padding:14px; margin-bottom:10px;">
        <div class="launch-status-label">${f.verification_status} · ${f.finding_id} · TOKEN ${f.token_id}</div>
        <div style="font-size:13px; color:var(--color-ink); margin-top:4px;">
          Hypothesis <span class="mono-code">${f.hypothesis_id}</span> confirmed by <strong>${escapeHtml(f.technician_id)}</strong> at ${f.confirmed_at_iso.slice(11, 19)} UTC
        </div>
        <div style="font-size:12px; color:var(--color-slate); margin-top:2px;">Observation: ${escapeHtml(f.technician_observation)}</div>
      </div>
    `
    )
    .join("");
}

async function loadIncidentView() {
  if (!state.activeCaseId) return;
  const { status, data } = await apiRequest("GET", `/api/cases/${state.activeCaseId}/incident`);
  if (status !== 200) return;
  state.incident = data;

  document.getElementById("incident-overview-container").innerHTML = `
    <div class="grid-4" style="margin-bottom: var(--spacing-28);">
      <div class="feature-card">
        <div class="launch-status-label">PEAK OBSERVED TEMP</div>
        <div class="card-metric-48" style="margin-top:8px;">${data.peak_observed_temp_c} °C</div>
        <p style="font-size:14px; color:var(--color-slate); margin-top:6px;">Critical threshold: ${data.critical_threshold_c} °C</p>
      </div>
      <div class="feature-card">
        <div class="launch-status-label">THRESHOLD BREACHED</div>
        <div class="card-metric-48" style="margin-top:8px;">${data.threshold_breached ? "YES" : "NO"}</div>
        <p style="font-size:14px; color:var(--color-slate); margin-top:6px;">Provenance: ${data.evidence_class}</p>
      </div>
      <div class="feature-card">
        <div class="launch-status-label">PREDICTION PATH</div>
        <div class="card-metric-48" style="margin-top:8px;">${data.prediction_type}</div>
        <p style="font-size:14px; color:var(--color-slate); margin-top:6px;">${(data.events || []).length} thermal regime events segmented</p>
      </div>
      <div class="feature-card">
        <div class="launch-status-label">VERIFIED FINDINGS</div>
        <div class="card-metric-48" style="margin-top:8px;">${data.confirmed_findings_count}</div>
        <p style="font-size:14px; color:var(--color-slate); margin-top:6px;">Append-only human confirmations</p>
      </div>
    </div>
  `;
}

async function runStep10Report(navigate = true) {
  if (!state.activeCaseId) return;
  const { status, data } = await apiRequest("GET", `/api/cases/${state.activeCaseId}/report`);
  if (status !== 200) {
    showToast(`Report locked: ${data.message}`);
    return;
  }
  const rep = data.report;
  state.report = rep;

  const confirmedHtml =
    (rep.confirmed_findings || []).length === 0
      ? `<p style="font-size:14px; color:var(--color-slate);">No findings confirmed by a technician yet. All hypotheses remain in Section 2 as unconfirmed drafts.</p>`
      : rep.confirmed_findings
          .map(
            (c) => `
          <div class="feature-card-bordered" style="padding:18px; margin-bottom:12px;">
            <div class="launch-status-label">CONFIRMED FINDING · ${c.verification_status} · SOURCE CLASS: ${c.source_evidence_class}</div>
            <h4 style="font-size:16px; font-weight:600; margin:6px 0;">Hypothesis ${c.hypothesis_id} (Check ${c.check_id})</h4>
            <p style="font-size:14px; color:var(--color-ink);">${escapeHtml(c.technician_observation)}</p>
            <div style="font-size:12px; color:var(--color-slate); margin-top:6px;">
              Technician: <span class="mono-code">${escapeHtml(c.technician_id)}</span> · Confirmed At: <span class="mono-code">${c.confirmed_at_iso}</span> · Token: <span class="mono-code">${c.token_id}</span>
            </div>
          </div>
        `
          )
          .join("");

  const draftsHtml =
    (rep.unconfirmed_drafts || []).length === 0
      ? `<p style="font-size:14px; color:var(--color-slate);">All diagnostic hypotheses have been adjudicated by a technician.</p>`
      : rep.unconfirmed_drafts
          .map(
            (d) => `
          <div class="feature-card-bordered" style="padding:18px; margin-bottom:12px;">
            <div class="launch-status-label">${d.watermark} · STATUS: ${d.verification_status}</div>
            <h4 style="font-size:16px; font-weight:600; margin:6px 0;">${escapeHtml(d.title)} (<span class="mono-code">${d.hypothesis_id}</span>)</h4>
            <p style="font-size:14px; color:var(--color-slate);">${escapeHtml(d.rendered_statement)}</p>
          </div>
        `
          )
          .join("");

  document.getElementById("report-view-container").innerHTML = `
    <div class="grid-2" style="margin-bottom: var(--spacing-28);">
      <div class="feature-card">
        <div class="launch-status-label">SECTION 1 · HUMAN-VERIFIED RECORD</div>
        <h3 class="card-title-24" style="margin-top:6px; margin-bottom:16px;">Technician-Confirmed Findings</h3>
        ${confirmedHtml}
      </div>
      <div class="feature-card">
        <div class="launch-status-label">SECTION 2 · UNVERIFIED DIAGNOSTIC DRAFTS</div>
        <h3 class="card-title-24" style="margin-top:6px; margin-bottom:16px;">Open Hypotheses (Not Confirmed Faults)</h3>
        ${draftsHtml}
      </div>
    </div>
    <div class="feature-card">
      <div class="launch-status-label">SECTION 3 · MARKDOWN AUDIT EXPORT &amp; KNOWN CONDITIONS</div>
      <pre style="font-family:var(--font-mono); font-size:12px; white-space:pre-wrap; background:var(--color-studio-mist); padding:20px; border-radius:18px; margin-top:12px; color:var(--color-ink);">${escapeHtml(rep.markdown_report)}</pre>
    </div>
  `;

  if (navigate) switchView("reports");
}

function downloadReportMarkdown() {
  if (!state.report || !state.report.markdown_report) {
    showToast("Generate the Step 10 report first.");
    return;
  }
  const blob = new Blob([state.report.markdown_report], { type: "text/markdown;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${state.activeCaseId || "coldsense"}-audit-report.md`;
  a.click();
  URL.revokeObjectURL(url);
}

async function runFullScientificPipeline() {
  showToast("Running full 10-step ColdSense scientific pipeline...");
  await launchDemoScenario("power_cut");
  await runStep2DataQuality(false);
  await runStep3Events(false);
  await runStep4Prediction(null, false);
  await runStep5Evidence(false);
  await runStep6NextBestCheck(false);
  await loadIncidentView();
  await runStep10Report(false);

  if (state.prediction && state.prediction.response_type === "SUCCESS") {
    document.getElementById("capsule-headline").textContent =
      `Active Case ${state.activeCaseId} (${state.activeCaseSummary.evidence_class}) · Fitted τ = ${state.prediction.time_constant_tau_hours} h (R² = ${state.prediction.r_squared})`;
    document.getElementById("capsule-subcopy").textContent =
      `All 10 workflow gates unlocked. Model RMSE ${state.prediction.thermal_model_rmse_c} °C vs. Naive Baseline ${state.prediction.naive_baseline_rmse_c} °C.`;
  }
  switchView("prediction");
  showToast("All 10 steps unlocked! Viewing Step 4 Thermal Drift Forecast.");
}

function escapeHtml(str) {
  return String(str || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

window.addEventListener("DOMContentLoaded", () => {
  initSessionAndDashboard();
});
