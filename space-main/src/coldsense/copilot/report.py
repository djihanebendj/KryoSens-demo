"""generate_service_report (docs/COPILOT.md §9, §8). The report is composed by the application from CONFIRMED
records only; drafts are listed as drafts and never as history. Technician statements are shown verbatim and
labelled USER_REPORTED (never restated as a measurement). The report text passes the same deterministic
numeric-claim check as a conversational answer: a number it contains must trace to a confirmed record or a case
tool result, or the report is refused with that number removed.
"""

from __future__ import annotations

from .numbers import Allowed, check


def _ident(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


def _allowed_for_report(ctx) -> Allowed:
    """Everything the application itself renders into the report is permitted: the confirmed statements
    (USER_REPORTED), their timestamps, the asset and model identifiers, and the app-generated record ids (list
    markers rendered by code, not the LLM). A number in a statement that cannot be parsed still fails closed."""
    allowed = Allowed()
    for r in ctx.findings.records(ctx.asset_id):
        allowed.add_text(r.statement)
        allowed.dates.add(r.created_at.date().isoformat())
        allowed.dates.add(r.confirmed_at.date().isoformat())
        allowed.identifiers.add(_ident(r.record_id))
    allowed.identifiers.add(_ident(ctx.asset_id))
    if ctx.model:
        allowed.identifiers.add(_ident(ctx.model))
    return allowed


def _diagnostic_section(ctx) -> str | None:
    """A service-report diagnostic section is app-rendered from the validated Phase 6 report. If a report is
    stored but no longer validates, the section states UNKNOWN rather than echoing any LLM prose."""
    report = getattr(ctx, "diagnostic", None)
    if report is None:
        return None
    from .diagnostics import DiagnosticValidationError, render_diagnostic, unknown_diagnostic
    try:
        return render_diagnostic(report)
    except DiagnosticValidationError:
        return unknown_diagnostic("the stored Phase 6 report failed application validation")


def build_report(ctx):
    from .tools import ToolResult, ToolStatus
    records = ctx.findings.records(ctx.asset_id)
    drafts = [d for d in ctx.findings.drafts() if d.asset_id == ctx.asset_id
              and not any(r.content_hash == d.content_hash for r in records)]
    lines = [f"SERVICE REPORT - asset {ctx.asset_id}" + (f" ({ctx.model})" if ctx.model else ""),
             "Confirmed technician findings (USER_REPORTED; outcomes default UNKNOWN):"]
    if records:
        for r in records:
            lines.append(f"  - [{r.record_id}] {r.created_at.date().isoformat()} "
                         f"(confirmed {r.confirmed_at.date().isoformat()} via {r.confirmed_via}; "
                         f"outcome {r.outcome.value}): \"{r.statement}\""
                         + (f" [window {r.event_window}]" if r.event_window else ""))
    else:
        lines.append("  (none confirmed)")
    if drafts:
        lines.append(f"Unconfirmed drafts, NOT part of history: {len(drafts)} (excluded from the report body)")
    diagnostic = _diagnostic_section(ctx)
    if diagnostic:
        lines += ["", diagnostic]
    text = "\n".join(lines)

    allowed = _allowed_for_report(ctx)
    allowed.quantities |= {(float(len(records)), ""), (float(len(drafts)), "")}   # app-rendered counts
    if diagnostic:
        allowed.add_text(diagnostic)                    # app-rendered from a validated report: trusted source
    violations = check(text, allowed)
    if violations:
        return ToolResult(tool="generate_service_report", status=ToolStatus.REFUSED,
                          summary="report refused: a value could not be traced to a confirmed record",
                          data={"violations": violations})
    return ToolResult(tool="generate_service_report", status=ToolStatus.OK,
                      summary=f"report with {len(records)} confirmed record(s), {len(drafts)} draft(s) excluded",
                      data={"report": text, "n_confirmed": len(records), "n_drafts_excluded": len(drafts)})
