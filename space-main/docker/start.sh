#!/bin/sh
# Start the API (one worker, loopback only), wait until it answers, then run the public Next.js server.
set -eu
cd /app
python -m uvicorn coldsense.app.api:create_app --factory --host 127.0.0.1 --port 8000 --workers 1 &
API_PID=$!
i=0
until python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=1)" 2>/dev/null; do
  i=$((i + 1))
  if ! kill -0 "$API_PID" 2>/dev/null; then echo "API failed to start (check COLDSENSE_API_* settings)" >&2; exit 1; fi
  if [ "$i" -ge 60 ]; then echo "API did not answer within 60 s" >&2; exit 1; fi
  sleep 1
done
cd /app/web
exec node server.js
