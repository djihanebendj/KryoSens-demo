#!/usr/bin/env bash
# Start the FastAPI service and the production Next.js server on localhost, run the browser end-to-end check,
# stop both. Run `npm run build` first. Usage: web/e2e/run-local.sh [screenshot-dir]
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
API_PORT="${API_PORT:-8000}"
WEB_PORT="${WEB_PORT:-3000}"
LOGS="$(mktemp -d)"

cleanup() { kill "${API_PID:-}" "${WEB_PID:-}" 2>/dev/null || true; }
trap cleanup EXIT

COLDSENSE_ENV=dev COLDSENSE_DATA_DIR="$(cd .. && pwd)/data" COLDSENSE_API_ALLOWED_ORIGINS="http://127.0.0.1:$WEB_PORT" \
  "$PY" -m uvicorn coldsense.app.api:create_app --factory --host 127.0.0.1 --port "$API_PORT" \
  >"$LOGS/api.log" 2>&1 &
API_PID=$!

# standalone output: static assets are served by the same server once copied next to it
rm -rf .next/standalone/.next/static && cp -r .next/static .next/standalone/.next/static
[ -d public ] && cp -r public .next/standalone/ 2>/dev/null || true
COLDSENSE_API_URL="http://127.0.0.1:$API_PORT" PORT="$WEB_PORT" HOSTNAME=127.0.0.1 NEXT_TELEMETRY_DISABLED=1 \
  node .next/standalone/server.js >"$LOGS/web.log" 2>&1 &
WEB_PID=$!

for _ in $(seq 1 60); do
  if curl -sf "http://127.0.0.1:$WEB_PORT/api/v1/health" >/dev/null; then break; fi
  sleep 0.5
done
curl -sf "http://127.0.0.1:$WEB_PORT/api/v1/health" || { echo "servers did not start; logs in $LOGS"; exit 1; }
echo
BASE_URL="http://127.0.0.1:$WEB_PORT" node e2e/local-e2e.cjs "${1:-}"
echo "server logs: $LOGS"
