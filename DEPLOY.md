# Deploying the experimental copilot to a Hugging Face Space

This Space is an **experiment**, separate from the main KryoSens demo. Nothing is hosted until the owner has
completed these steps, opened the public URL and used it.

## Owner steps

1. On huggingface.co: **New Space** → SDK **Docker** → template **Blank** → visibility **Public** → Create.
2. **Upload the files** from `space-copilot.zip` (unzipped) to the Space root: `Dockerfile`, `.dockerignore`,
   `README.md` (it holds the Space YAML header), `DEPLOY.md`, `run_server.py`, `src/`. Do not upload any `.env`.
3. Settings → **Variables and secrets**, then add:

   | Kind | Name | Value |
   |---|---|---|
   | Secret | `GEMINI_API_KEY` | a **new** key created in a **new** Google AI Studio project (never a key that was shared in chat) |
   | Secret | `COLDSENSE_SESSION_SECRET` | random, 32+ characters, e.g. `python -c "import secrets; print(secrets.token_urlsafe(48))"` |
   | Secret | `COLDSENSE_CONFIRMATION_SECRET` | another random value, 32+ characters |
   | Variable | `COLDSENSE_COPILOT_ENABLED` | `true` |
   | Variable | `GEMINI_MODEL` | `gemini-3-flash-preview` |
   | Variable | `GEMINI_THINKING_LEVEL` | `low` |
   | Variable | `COLDSENSE_ALLOWED_ORIGINS` | `https://<owner>-<space>.hf.space` (all lowercase, no trailing slash) |

4. Wait for the build, then open the **direct** `https://<owner>-<space>.hf.space` URL (not the huggingface.co
   page), so that the browser's Origin matches `COLDSENSE_ALLOWED_ORIGINS`.
5. Ask once: "What does alarm E3 mean on the EX-200?" and read the **engine** label on the answer: `GEMINI` while the
   free quota lasts, otherwise `EXTRACTIVE_FALLBACK` with a visible notice. Ask sparingly; the free tier is about
   20 requests per day.

## Behaviour to know

- **Origins:** writes need a signed session cookie, a CSRF token and an allowed `Origin`. The defaults are the
  localhost origins; `COLDSENSE_ALLOWED_ORIGINS` (comma-separated) **adds** origins. If it is unset, every POST from
  the Space URL is rejected with 403.
- **Secrets:** `COLDSENSE_SESSION_SECRET` and `COLDSENSE_CONFIRMATION_SECRET` are read from the environment. If
  either is missing, the server generates a random one and logs a warning (the value itself is never printed).
  Sessions and confirmations then become invalid on every restart, so set both.
- **Copilot:** off unless `COLDSENSE_COPILOT_ENABLED=true` in the server environment; no page can switch it on.
  Without `GEMINI_API_KEY`, or once the quota is exhausted, answers are extractive quotes from the corpus.
- **State:** kept in memory; a sleeping or restarted Space loses all cases.
- **Corpus:** synthetic maintenance documents only; not manufacturer documentation.
