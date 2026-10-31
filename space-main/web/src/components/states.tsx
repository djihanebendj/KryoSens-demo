import Link from "next/link";

import type { ApiFailure } from "@/lib/api/client";

export function EmptyState({ title, children, action }: { title: string; children?: React.ReactNode; action?: { href: string; label: string } }) {
  return (
    <section role="status" data-state="empty" className="rounded-lg border border-dashed border-slate-300 bg-white p-8 text-center">
      <h2 className="text-lg font-semibold text-slate-800">{title}</h2>
      {children && <div className="mt-2 text-sm text-slate-600">{children}</div>}
      {action && (
        <Link href={action.href} className="mt-4 inline-block rounded-md bg-sky-700 px-4 py-2 text-sm font-medium text-white hover:bg-sky-800">
          {action.label}
        </Link>
      )}
    </section>
  );
}

export function LoadingState({ label }: { label: string }) {
  return (
    <div role="status" aria-live="polite" data-state="loading" className="flex items-center gap-3 rounded-lg border border-slate-200 bg-white p-6 text-sm text-slate-600">
      <span className="h-4 w-4 animate-spin rounded-full border-2 border-slate-300 border-t-sky-700" aria-hidden />
      {label}
    </div>
  );
}

const HINTS: Partial<Record<string, string>> = {
  CASE_EXPIRED: "Cases are deleted after their lifetime. Upload the file again to start a new case.",
  CASE_NOT_FOUND: "This case is not available in this browser session. Upload the file again.",
  SESSION_EXPIRED: "Your session expired. Upload the file again to start a new case.",
  SESSION_INVALID: "Your session is not valid for this server. Upload the file again.",
  SESSION_REQUIRED: "Upload a file to start a case.",
  BACKEND_UNAVAILABLE: "The analysis service is not reachable. Start the API (scripts/dev.ps1) or check COLDSENSE_API_URL in web/.env.local.",
  NETWORK_ERROR: "The browser could not reach this web application. Check that the frontend is running.",
  BUSY: "The service is busy with other analyses. Try again in a moment.",
  RATE_LIMITED: "Too many requests from this browser in a short time. Wait a moment and try again.",
  REQUEST_TOO_LARGE: "The request body is larger than this service accepts (10 MiB upload limit).",
  UPLOAD_TOO_LARGE: "The file is larger than the 10 MiB upload limit (or exceeds 200,000 rows).",
  REQUEST_TIMEOUT: "The service stopped a request that took too long. Nothing was written by that request.",
  REPORT_NOT_GENERATED: "Generate a report for this case first; there is no latest report yet.",
  FINDING_DRAFT_NOT_FOUND: "This draft is no longer available for this case. Prepare a new draft.",
  FINDING_NOT_READY: "This draft has no live confirmation token. Prepare it again.",
  FINDING_CANCELLED: "This draft was cancelled. Nothing was recorded and it cannot be confirmed.",
  FINDING_ALREADY_CONFIRMED: "This draft is already confirmed. Refresh the incident to see the record.",
  FINDING_CONFIRMATION_INVALID: "The confirmation token does not match this draft. Prepare it again.",
  FINDING_CONFIRMATION_EXPIRED: "The single-use confirmation token expired. Prepare the draft again; nothing was written.",
  FINDING_CONFIRMATION_REPLAY: "That confirmation was already used. Nothing was written twice; refresh the incident.",
};

export function ErrorState({ error, onRetry }: { error: ApiFailure; onRetry?: () => void }) {
  const hint = HINTS[error.code];
  return (
    <section role="alert" data-state="error" data-code={error.code} className="rounded-lg border border-rose-300 bg-rose-50 p-5 text-sm">
      <p className="font-semibold text-rose-900">
        Error <code className="rounded bg-rose-100 px-1">{error.code}</code>
      </p>
      <p className="mt-1 text-rose-900">{error.message}</p>
      {hint && <p className="mt-2 text-rose-800">{hint}</p>}
      <div className="mt-3 flex gap-3">
        {onRetry && error.body.retryable && (
          <button type="button" onClick={onRetry} className="rounded-md border border-rose-400 px-3 py-1 text-rose-900 hover:bg-rose-100">
            Try again
          </button>
        )}
        {(hint && error.code.startsWith("SESSION")) || error.code.startsWith("CASE") ? (
          <Link href="/upload" className="rounded-md bg-rose-700 px-3 py-1 text-white hover:bg-rose-800">
            Upload data
          </Link>
        ) : null}
      </div>
    </section>
  );
}

export function NotAvailable({ title, stage, children }: { title: string; stage: string; children?: React.ReactNode }) {
  return (
    <section data-state="not-available" className="rounded-lg border border-slate-200 bg-white p-6">
      <h1 className="text-xl font-semibold text-slate-900">{title}</h1>
      <p className="mt-3 inline-block rounded bg-slate-100 px-2 py-1 font-mono text-xs text-slate-700">NOT_AVAILABLE</p>
      <p className="mt-3 text-sm text-slate-600">
        This page is not available in this release ({stage}). Nothing on it is computed or simulated.
      </p>
      {children && <div className="mt-2 text-sm text-slate-600">{children}</div>}
    </section>
  );
}
