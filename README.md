---
title: KryoSens Experimental Copilot
emoji: ❄️
sdk: docker
app_port: 7860
pinned: false
---

> **Experimental Space, not the main KryoSens product.** A rule-constrained, grounded retrieval (RAG) copilot over
> synthetic maintenance documents, built by Aiden8teen with the team's security fixes: a hazard gate runs before
> any model call, every number must come from the retrieved passage, and answers carry verbatim quotes. It calls
> Gemini only while the free daily quota lasts (about 20 requests/day); after that it answers with grounded
> extractive quotes and shows a visible notice. Data lives in memory and is lost on restart. Deployment steps:
> `DEPLOY.md`.

# KryoSens — cold-room decision support (merged build)

Upload a temperature log from a cold room, get a data-quality verdict, an event timeline, a
no-cooling forecast or a stated refusal, cautious hypotheses with the next check to perform, and a
report built only from findings a technician confirmed. A retrieval copilot answers questions from a
document corpus with verbatim quotes, behind a hazard gate and a numeric-integrity audit.

**What this build is:** a local demo, hardened. State lives in server memory, cases expire after one
hour, and there is no deployment. Read "Honest status" before describing it to anyone.

## Honest status

| Area | State |
|---|---|
| Ingestion, data quality, events, thermal prediction, evidence, next check, findings, report | implemented and tested, on simulated and uploaded logs |
| Uploads | always evidence class `UNVALIDATED`; an uploader's "this is real" claim is stored separately and never upgrades it |
| Hypotheses | always rendered as `HYPOTHESIS [Unconfirmed]`; nothing is declared a confirmed fault |
| Hazard gate | blocks before any model call; verified to make zero model calls |
| Numeric integrity | every number in an answer must come from the retrieved passages; one retry, then a refusal |
| Copilot (Gemini) | **off by default.** Wired and unit-tested; the live path reached `engine: GEMINI` once, then the free-tier daily quota (20 requests/model/day) was exhausted, so the fixed configuration is **unverified end to end** |
| Validation on real cold rooms | none. No forecast accuracy on real data is claimed |
| Arabic | not supported; non-Latin script is refused before any model call |
| Persistence, deployment, CI | none |

## Install and run

```bash
python3 -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
python -m pytest -q                                # 48 passed, 1 skipped
python run_server.py                               # http://127.0.0.1:8000
```

The start-up line states the copilot's resolved state, for example
`Copilot: DISABLED (COLDSENSE_COPILOT_ENABLED, default off)`.

Optional extras: `pip install -e ".[rag]"` adds LangChain and LangGraph, `".[api]"` adds FastAPI.
Neither is needed for the server above, which uses only the standard library.

## Configuration

Copy `.env.example` to `.env` and fill it in. `.env` is git-ignored; never commit a key.

| Variable | Default | Meaning |
|---|---|---|
| `COLDSENSE_COPILOT_ENABLED` | `false` | The only switch for the copilot. No route and no browser action can change it |
| `GEMINI_API_KEY` | unset | Without it the copilot answers from the deterministic extractive synthesizer |
| `GEMINI_MODEL` | `gemini-3-flash-preview` | `gemini-2.5-flash` is retired and returns HTTP 404 |
| `GEMINI_THINKING_LEVEL` | `low` | Reasoning effort. Gemini 3 spends tokens thinking before it answers; "low" leaves room for the answer |
| `GEMINI_EMBEDDING_MODEL` | `gemini-embedding-001` | Verified working (HTTP 200) |
| `COLDSENSE_SESSION_SECRET` | random per process | Signs session cookies and CSRF tokens. Unset means cookies break on every restart, and the server warns |
| `COLDSENSE_CONFIRMATION_SECRET` | random per process | Signs technician confirmation tokens, same caveat |

## Confirming the copilot really uses Gemini

```bash
# .env: GEMINI_API_KEY=<your key>, COLDSENSE_COPILOT_ENABLED=true, GEMINI_MODEL=gemini-3-flash-preview
python run_server.py
# open http://127.0.0.1:8000 -> Technician Copilot -> ask:
#   What does alarm E3 mean on the EX-200?
```

The banner above the answer reads `ENGINE: GEMINI` or `ENGINE: EXTRACTIVE_FALLBACK`, and a fallback
always shows why. `Reason: Gemini REST call failed: 429` means the daily free-tier quota is spent;
retry after the window stated by the API (it reports a `retryDelay`).

## Security

Session cookies are opaque, signed and HttpOnly. Every state-changing request needs an allowed
`Origin` and a CSRF token bound to the session. A case belongs to one visitor; another visitor gets
the same `CASE_NOT_FOUND` reply whether the case exists or not. Uploads are capped at 10 MiB and
200,000 rows, with `Content-Length` checked before the body is read. Every response carries
`X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy` and a Content-Security-Policy. All SQL
uses bound parameters. The API key travels in the `x-goog-api-key` header, never in a URL.

## Tests

`python -m pytest -q` → **48 passed, 1 skipped**. The skip is a test documenting the `.env` contract,
which skips when no key is configured. `tests/conftest.py` makes the suite hermetic: it clears
`GEMINI_*` and `COLDSENSE_*` and neutralises the `.env` reader, so a developer's key cannot change a
result or trigger a live call.

## Known limits

- The retriever answers questions that ask for a quantity or a named identifier. Questions phrased as
  "how do I…" or "why is…" return `NO_RELEVANT_DOCUMENTATION` even when the corpus covers the topic
  (measured: 6 of 10 realistic questions answered, 3 refused although covered, 1 correctly refused).
- The document corpus is synthetic: fictional products (EX-200, EX-220, TF-90, CU-40).
- `pyproject.toml` declares `readme = "PRD.md"`, a file that does not exist; a wheel build would fail.
- One hour case lifetime, in-memory only, single process.

## Do not claim

Not "validated on real data", not "real forecast accuracy", not "Arabic support", not "door, defrost
or power logs as inputs". The copilot is **experimental, Gemini-based and constrained by rules** —
nothing more.

See `MERGE_NOTES.md` for the merge history and what changed from the upstream base.
