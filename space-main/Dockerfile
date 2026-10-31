# syntax=docker/dockerfile:1
# KryoSens hosted DEMO: one container, one public port (7860, Hugging Face Spaces default).
#   public  :7860  Next.js production server (UI + same-origin /api/v1/* proxy)
#   private :8000  FastAPI, ONE uvicorn worker (cases and sessions live in memory), bound to 127.0.0.1
# The copilot stays OFF (FEATURE_DISABLED). No secret is baked in: set them as Space secrets (docs/DEPLOY.md).

FROM node:22-bookworm-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
# Optional build secret "ca": an extra CA for builds behind a TLS-inspecting proxy (unused on Spaces/Render).
RUN --mount=type=secret,id=ca,required=false \
    if [ -f /run/secrets/ca ]; then export NODE_EXTRA_CA_CERTS=/run/secrets/ca; fi; \
    npx --yes npm@11.6.2 ci --no-audit --no-fund
COPY web/ ./
ENV NEXT_TELEMETRY_DISABLED=1
RUN npm run build

FROM python:3.13-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 NEXT_TELEMETRY_DISABLED=1 \
    NODE_ENV=production COLDSENSE_ENV=prod PORT=7860 HOSTNAME=0.0.0.0 \
    COLDSENSE_API_URL=http://127.0.0.1:8000
COPY --from=web /usr/local/bin/node /usr/local/bin/node
RUN useradd --create-home --uid 1000 app
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ src/
RUN --mount=type=secret,id=ca,required=false \
    if [ -f /run/secrets/ca ]; then export PIP_CERT=/run/secrets/ca; fi; \
    pip install ".[api]"
COPY data/sample/ data/sample/
COPY --from=web /web/.next/standalone/ web/
COPY --from=web /web/.next/static/ web/.next/static/
COPY docker/start.sh /app/start.sh
RUN chmod 755 /app/start.sh && chown -R app:app /app
USER app
EXPOSE 7860
CMD ["/app/start.sh"]
