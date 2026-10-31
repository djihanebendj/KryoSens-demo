// Same-origin proxy: /api/v1/* -> the FastAPI service (COLDSENSE_API_URL, server-side only).
// The browser only ever talks to this origin, so the session cookie is first-party and no CORS is needed.
// Only an allowlist of headers crosses in either direction.
import type { NextRequest } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const BACKEND = (process.env.COLDSENSE_API_URL ?? "http://127.0.0.1:8000").replace(/\/+$/, "");
const UPLOAD_MAX_BYTES = 10 * 1024 * 1024;
const BODY_SLACK = 64 * 1024; // the backend enforces the exact limit; this only stops obvious abuse early
const TIMEOUT_MS = 90_000; // a forecast with the reliability assessment takes seconds; queueing adds up to 10 s
// origin + x-csrf-token: the API validates both on every state-changing request (amendment G)
const REQUEST_HEADERS = ["content-type", "content-length", "cookie", "accept", "origin", "x-csrf-token"];
const RESPONSE_HEADERS = ["content-type", "cache-control"];

function envelope(status: number, code: string, message: string, retryable = false) {
  return Response.json({ error: { code, message, retryable } }, { status, headers: { "Cache-Control": "no-store" } });
}

async function forward(req: NextRequest, ctx: { params: Promise<{ path: string[] }> }) {
  const { path } = await ctx.params;
  if (path[0] !== "v1" || path.some((p) => p === ".." || p === "." || p.includes("\\"))) {
    return envelope(404, "NOT_AVAILABLE", "no such endpoint");
  }
  const declared = Number(req.headers.get("content-length") ?? "0");
  if (declared > UPLOAD_MAX_BYTES + BODY_SLACK) {
    return envelope(413, "UPLOAD_TOO_LARGE", `the file exceeds ${UPLOAD_MAX_BYTES} bytes`);
  }
  const headers = new Headers();
  for (const h of REQUEST_HEADERS) {
    const v = req.headers.get(h);
    if (v !== null) headers.set(h, v);
  }
  const target = `${BACKEND}/api/${path.map(encodeURIComponent).join("/")}${req.nextUrl.search}`;
  const hasBody = !["GET", "HEAD"].includes(req.method);
  let upstream: Response;
  try {
    upstream = await fetch(target, {
      method: req.method,
      headers,
      body: hasBody ? req.body : undefined,
      // @ts-expect-error -- Node's fetch needs duplex for a streamed request body; not in the DOM RequestInit type
      duplex: hasBody ? "half" : undefined,
      redirect: "manual",
      cache: "no-store",
      signal: AbortSignal.timeout(TIMEOUT_MS),
    });
  } catch (e) {
    if (e instanceof Error && e.name === "TimeoutError") {
      return envelope(504, "REQUEST_TIMEOUT",
        "the analysis service did not answer within 90 s; it may still finish the work, so refresh before retrying", true);
    }
    return envelope(502, "BACKEND_UNAVAILABLE", "the analysis service is not reachable; try again shortly", true);
  }
  const out = new Headers();
  for (const h of RESPONSE_HEADERS) {
    const v = upstream.headers.get(h);
    if (v !== null) out.set(h, v);
  }
  for (const c of upstream.headers.getSetCookie()) out.append("set-cookie", c);
  return new Response(upstream.status === 204 ? null : upstream.body, { status: upstream.status, headers: out });
}

export { forward as GET, forward as POST, forward as DELETE };
