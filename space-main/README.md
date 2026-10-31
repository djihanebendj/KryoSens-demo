---
title: KryoSens Demo
emoji: ❄️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---
# KryoSens

Evidence-first decision support for refrigeration. KryoSens (code package: `coldsense`) reads a cold
room's own temperature log, decides whether the data is good enough to use,
learns how fast that room warms when cooling stops, and estimates how long it
would stay below a temperature limit if cooling were lost. When the evidence
does not support an answer, it refuses and says why.

> **Status — Phase 9: local end-to-end web application (Stage 3 implemented locally; NOT DEPLOYED, no public URL).** Phases 1–4 accepted; Phases 5 and 6 accepted with conditions (5b1a220, 997de5d); Phase 7 conditionally complete (an opt-in local semantic model exists, but semantic refusal is unreliable; see docs/KNOWLEDGE.md §1); Phase 8 hardening merged on `integration/final`, tested with a fake LLM only and with no formal acceptance record; Phase 10 not started.
> No result in this repository has been validated on a physical cold room, and
> no real logger export has been processed yet.
> All numbers produced so far come from `SIMULATED` samples or `UNVALIDATED` uploads; none is a validated
> measurement. The Copilot is disabled and makes no LLM call. See `docs/PHASE9_LIMITATIONS.md` §0.

## What KryoSens is

A local web application and CLI that reads a cold room's temperature log, says whether the file can be
analysed, splits it into events, forecasts how the room warms with no active cooling (or refuses and says
why), lists cautious candidate explanations of each excursion, and ranks the single check worth doing next.
A technician confirms findings; the report contains confirmed findings only. Nothing is ever reported as a
confirmed fault by the software.

## Run it (clean clone)

```bash
git clone <repo> kryosens && cd kryosens
python3 -m venv .venv && . .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[api,dev]"
# backend: one worker, state lives in memory
COLDSENSE_ENV=dev COLDSENSE_DATA_DIR="$PWD/data" COLDSENSE_API_ALLOWED_ORIGINS=http://127.0.0.1:3000 \
  python -m uvicorn coldsense.app.api:create_app --factory --host 127.0.0.1 --port 8000
# frontend, in a second terminal (npm 11 is required: the lockfile is npm-11 format)
cd web && npx npm@11.6.2 ci && npm run build && node .next/standalone/server.js     # http://127.0.0.1:3000
```

Tests: `python -m pytest -q` (backend), `cd web && npx vitest run && npm run lint && npm run typecheck`,
and after a build `bash web/e2e/run-local.sh` (starts both servers itself).

## Click path (10 steps)

1. `/` landing page → **Try the demo (simulated log)** creates a case from the bundled SIMULATED sample.
2. **Data Quality** (`/quality`): the verdict and every issue found.
3. **Temperature Timeline** (`/timeline`): set a limit (e.g. upper 8 °C) and run the analysis.
4. **Prediction** (`/prediction`): run the no-cooling forecast, or read its refusal.
5. **Evidence** (`/evidence`): facts, model inferences and recommendations kept apart.
6. **Next Best Check** (`/next-check`): the ranked checks and why each is useful.
7. **Investigation** (`/investigation`): the PROPOSED hypotheses and their evidence.
8. **Incident** (`/incident`): write a finding, review the exact draft, confirm or cancel.
9. **Reports** (`/reports`): compose the report from confirmed findings only.
10. **Technician Copilot** (`/copilot`): the free-text copilot is off; **Guided Start** shows the backend's
    own next check and hypotheses, rule-based, with no language model.
The previous home page is at `/dashboard`.

## Limitations

- No validation on a real cold room; no claim of real forecast accuracy.
- The only equipment signal the web app reads is the `compressor_state` column (it also reads timestamp,
  temperature and an optional ambient temperature). Door, defrost and power records are a CLI input only.
- No Arabic support: the interface and messages are English only.
- Case data lives in server memory and is deleted after 2 hours; a restart loses it.
- Guided Start is rule-based: it displays backend output and generates no text.
- The free-text copilot is off in this app: the server gate always answers `FEATURE_DISABLED`.

## Experimental copilot

A separate, experimental, rule-constrained, Gemini-based copilot with grounded retrieval, by Aiden8teen
(repository kyrosync) with security fixes by the team, is in `experiments/copilot-kyrosync/`. It is **not
integrated** into this app and is excluded from this repository's tests and linting. See
`experiments/copilot-kyrosync/README_EXPERIMENT.md`.

## Security notes

Not a penetration test. What the app does: opaque HttpOnly session cookies; an Origin allowlist and a
signed double-submit CSRF token on every state-changing request; a case is visible to its owner only;
uploads capped at 10 MiB and 200,000 rows; every upload analysed as `UNVALIDATED` whatever the uploader
declares; security headers on pages (CSP with `frame-ancestors 'none'`, `X-Frame-Options: DENY`). No
secret is stored in the repository.

## Progress

| Work | Status |
|---|---|
| Cold-room data foundation: ingest, data-quality gate, warming segments, naive baseline with refusal | done, SIMULATED data only |
| **Phase 1 — Architecture and Project Foundation:** src layout, core types, settings, logging, error handling, CLI, architecture tests | done |
| **Phase 2 — Data Ingestion and Data Validation:** traceable observations table, explicit timestamp/unit/number parsing, row flags, structured quality report, `coldsense validate` | accepted; SIMULATED and hand-written test files only |
| **Phase 3 — Event Segmentation:** gaps, stable/rising/falling/irregular periods, threshold excursions, compressor transitions and cycles, all traceable to rows; `coldsense events` | accepted (checkpoint 101ca5f); SIMULATED data only, provisional parameters |
| **Phase 4 — Physics-aware thermal modelling and forecasting:** input inventory, regime evidence, first-order models (fitted / measured / assumed equilibrium, time-varying ambient, active cooling as hindcast), identifiability checks, persistence and recent-slope baselines, held-out comparison, threshold crossings, structured refusals; `coldsense forecast`, `coldsense thermal-models` | accepted (checkpoint 8687272); SIMULATED data only, provisional limits |
| **Phase 5 — Uncertainty, calibration and reliability gate:** causal view (log cut at the origin), empirical intervals from held-out forecasts with a separate coverage check, horizon diagnostics, parameter envelopes, crossing-time intervals, rule-based reliability (HIGH / MEDIUM / LOW / UNSUPPORTED, not a probability); `coldsense forecast --reliability` | accepted with conditions (5b1a220); SIMULATED data only, provisional limits |
| **Phase 6 — Evidence, diagnostics and next-best checks:** traceable observed facts, rule-based assessment of candidate explanations of sustained excursions against recorded evidence only, PROPOSED hypotheses (no probabilities), ranked next-best checks, INSUFFICIENT_EVIDENCE when nothing recorded bears on them; `coldsense diagnose` | accepted with conditions (997de5d); SIMULATED data only, rules UNVALIDATED |
| **Phase 7 — Technical knowledge and source-traceable retrieval:** local ingestion of TXT, Markdown, DOCX (PDF with the optional `pypdf`), screening of untrusted text, section-aware chunks with pages/lines and revisions, lexical embeddings in a SQLite store, cited extractive answers or NO_RELEVANT_DOCUMENTATION; `coldsense kb` | **CONDITIONALLY COMPLETE**: the embedding stage is lexical, not semantic, and retrieval decisions do not use vector similarity (docs/KNOWLEDGE.md §1); evaluated on a small SYNTHETIC corpus only |
| **Phase 8 — Tool calling and technician copilot** | hardening merged on `integration/final`; tested with a fake LLM only; no formal acceptance record and no recorded mutation results (see docs/COPILOT.md) |
| **Phase 9 — Next.js frontend** (local end-to-end app: FastAPI service, upload/quality/prediction, analysis workspace, technician findings, reports, disabled Copilot, public-demo hardening) | **Stage 3 implemented locally; not deployed**; see `docs/PHASE9_API.md`, `docs/PHASE9_DEPLOYMENT.md`, `docs/PHASE9_LIMITATIONS.md` |
| Phase 10 (deployment) | not started |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest -q
coldsense validate data/sample/SIMULATED_coldroom_24h_1min.csv --evidence SIMULATED
coldsense events data/sample/SIMULATED_coldroom_24h_1min.csv --evidence SIMULATED --upper-threshold 5
coldsense analyze data/sample/SIMULATED_coldroom_24h_1min.csv --threshold 8 --now 3.5 --evidence SIMULATED
coldsense thermal-models
python -m coldsense.domains.cold_room.thermal.benchmark   # SIMULATED ground-truth benchmark
python -m coldsense.domains.cold_room.thermal.reliability_benchmark --rooms 20   # Phase 5, SIMULATED rooms
```

`coldsense forecast LOG.csv --horizon-min 60 [--threshold 8]` needs evidence of a
period without active cooling (a `compressor_state` column, `--passive-since`, or a
`POWER_INTERRUPTION` record in `--events EVENTS.csv`); on the committed sample (no
compressor column) it refuses and says what would allow it. `--question continuation`
asks instead whether the current behaviour continues (baselines only from a cycling
origin). `--metadata ROOM.json` is validated and disclosed, not used in calculations;
`--temperature-column` picks one sensor of several. Every result lists available and
missing inputs, why the model was chosen over more informative ones, and ranked
missing-input recommendations. Details: `docs/THERMAL_MODELS.md` §14.

`--reliability` adds the Phase 5 assessment: the same forecast computed from the log
as it was at the origin, an empirical interval only when held-out coverage supports
it, and a reliability status with every rule that passed or failed. Each passive period
of the log counts as one observation (not each forecast case). It is a rule-based
classification, not a probability; REAL logs are capped at MEDIUM until the method has
been validated against real outcomes, and EXPECTED or UNVALIDATED (unlabelled) inputs
are never treated as real measurements. Details: `docs/RELIABILITY.md`.

`coldsense diagnose LOG.csv --upper-threshold 8 [--events EVENTS.csv] [--metadata ROOM.json]` lists observed
facts with their source lines, assesses each candidate explanation of a sustained excursion against recorded
evidence only (compressor state, supplied records, validation flags), and ranks next-best checks (NEXT BEST
CHECK, with WHY, EVIDENCE and EXPECTED INFORMATION). It gives no probability and names no cause; without a limit, or when nothing recorded bears on any explanation, it returns
INSUFFICIENT_EVIDENCE (exit 2). `--horizon-min` adds the Phase 4/5 result as context only. Details:
`docs/DIAGNOSTICS.md`.

`coldsense kb ingest MANUAL.pdf --model EX-200 --revision B` indexes a document locally (metadata is never
inferred); `coldsense kb query "defrost interval EX-200"` returns cited passages and verbatim quotes, shows
disagreeing revisions side by side, or answers NO_RELEVANT_DOCUMENTATION (exit 2). Retrieval is lexical: the
question must use the documentation's words. `coldsense kb evaluate` runs the synthetic evaluation. Details:
`docs/KNOWLEDGE.md`.

Other commands: `coldsense modules`, `coldsense doctor`, `coldsense version`
(or `python -m coldsense ...`). Exit codes: 0 ok, 1 error, 2 refused (data not
suitable, or prediction refused).

`validate` prints the data-quality report; `--json` prints it in full and
`--export DIR` also saves every input row with its status and flags (never
overwrites). Input options for both commands: `--unit C|F|K`, `--timezone`,
`--timestamp-format`, `--delimiter`, `--encoding` (e.g. `--encoding cp1256`
for Arabic Windows exports; without it, non-UTF-8 files are read as cp1252 with
a warning). Validation thresholds are provisional and have not been checked
against real logger exports.

Input CSV: a timestamp and a temperature column are required; ambient
temperature and compressor state are optional. Common English and French
headers, `;` delimiters and decimal commas are recognised. Rows are never
deleted: unusable ones are kept, flagged and excluded from analysis; rows with
too many fields are kept as raw text with their line number. Ambiguous
dates, unknown units and missing time zones are reported, not guessed. Full
rules: `docs/DATA_VALIDATION.md`. Event segmentation: `docs/EVENTS.md`. Thermal models: `docs/THERMAL_MODELS.md`.

## Web app (Phase 9, local only)

```powershell
# one-time setup
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[api,dev]"
cd web; npm ci; cd ..

# one command starts the API and the web app, waits for health, and prints the URLs
powershell -File scripts/dev.ps1
# then in another terminal:
powershell -File scripts/smoke_test.ps1
```

The browser talks only to the Next.js origin; `/api/v1/*` is proxied server-side. The proxy reads
`COLDSENSE_API_URL` from `web/.env.local` (copy `web/.env.example`); it is not a `NEXT_PUBLIC_*` value, so the
backend URL is not exposed to browser code. Use `data/sample/SIMULATED_power_cut_22h_1min.csv` (forecast; horizon
120, threshold 8 for a visible excursion) and `data/sample/SIMULATED_coldroom_24h_1min.csv` (refusal: no compressor
state). The smoke test copies the first sample to `data/cold_room_test.csv` and walks the whole flow.

Build the Phase 7 SQLite knowledge index once; it powers the read-only documentation search on the Evidence page:

```powershell
powershell -File scripts/index_knowledge.ps1
& .\.venv\Scripts\python.exe -m coldsense.app.main kb query "EX-200 high-temperature alarm delay" --store data\knowledge\kb.sqlite
```

The index is built from the repository's SYNTHETIC, fictional evaluation corpus by default; replace it with your own
manuals if you have them. The lexical provider is local and needs no API key. This is the only database in the local
stack: a case and its results live in memory for 2 hours, and a restart loses them.

Workflow: upload/sample → Data Quality → Temperature Timeline → Prediction → Evidence / Next Best Check /
Investigation → **Incident** (type a finding, review the exact draft, choose an outcome, confirm or cancel) →
**Reports** (confirmed records only; drafts excluded; a numeric-check refusal has no report text) → **Technician
Copilot** (free-text assistant: server-side `FEATURE_DISABLED`, no LLM call; Guided Start: a rule-based,
read-only view of the backend's top next check, PROPOSED hypotheses and missing evidence, with no AI text and no
repair procedures). A case lives 2 hours in memory; a restart loses it.

No public deployment and no public URL exist. The free-text Copilot is disabled: no verified Phase 8 Rule 7 acceptance
record exists, and this slice does not wire a conversation provider. The next step for it is a real language model
plus a formal Phase 8 acceptance record. API contract, deployment/env notes, the user workflow
and the exact limitations are in `docs/PHASE9_API.md`, `docs/PHASE9_DEPLOYMENT.md` and
`docs/PHASE9_LIMITATIONS.md`; architecture and hosting notes remain in `docs/PHASE9_ARCHITECTURE.md`.

## Evidence classes

| Class | Meaning |
|---|---|
| `REAL` | measured on a physical unit |
| `SIMULATED` | produced by the simulator |
| `EXPECTED` | a model output not yet checked against a real outcome |
| `UNVALIDATED` | data of unknown provenance (default for uploads) |

A prediction from `REAL` data is still `EXPECTED` until an unplug test confirms it.

## When ColdSense refuses

| Condition | Default limit (`domains/cold_room/thresholds.py`) |
|---|---|
| Log shorter than | 6 h (24 h recommended) |
| Sampling coarser than | 10 min |
| Coverage below | 80 % |
| Clean warming segments fewer than | 5 |
| Warming-rate variation (CV) above | 0.5 |
| Threshold beyond warmest observed off-cycle temperature by more than | 5 °C |

## Known limitations (measured on simulated data)

1. **The baseline errs early far from the setpoint.** On the simulated room,
   3.5 → 10 °C is predicted at 91 min against a true 108 min (−16 %).
2. **The range is not a confidence interval.** It reflects cycle-to-cycle
   spread only and misses extrapolation error; a test pins this.
3. **Frequent door openings starve the model.** At ~40 openings/day only 3
   usable off-cycles remain and ColdSense refuses (`docs/RESEARCH_LOG.md`, H-001).
4. Compressor state is read from the log's `compressor_state` column when present and is never
   inferred. Air temperature at one sensor is not product-core temperature. ColdSense makes no
   food-safety claim.

## Repository layout

```
src/coldsense/core/               evidence classes, results, errors, governance, module registry
src/coldsense/domains/cold_room/  the cold-room module
src/coldsense/app/                command-line entry point
tests/                            platform and module tests
data/sample/                      SIMULATED_* files only; real data is never committed by default
docs/                             ARCHITECTURE.md, DECISIONS.md, RESEARCH_LOG.md
```

Configuration: copy `.env.example` to `.env` to change log level, log format
or data folders. Refusal thresholds are not configurable by environment, by design.

## Team

| Member | Role |
|---|---|
| Mohamed | product and coordination |
| Tinhinane | design and presentation (landing page) |
| Meriem | QA |
| Djihane | Windows runs |
| Aiden8teen | experimental copilot and RAG (kyrosync) |

## Tech stack

Python 3.11+, pandas, NumPy, SciPy, Pydantic, pydantic-settings, pytest, ruff.
AI assistance (Claude) was used for design and code generation; all code is
reviewed and tested by the team.
