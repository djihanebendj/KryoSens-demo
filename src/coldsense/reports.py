"""Step 10: Incident & Diagnostic Report Builder.

Enforces:
- Strictly separates Technician-Confirmed Findings from Unconfirmed Drafts/Hypotheses.
- Never upgrades UNVALIDATED or SIMULATED data to REAL.
- Explicitly states active conditions, refusal status, and known limitations (No Overclaiming).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

from coldsense.models import (
    CaseRecord,
    PredictionRefusal,
    PredictionSuccess,
    VerificationStatus,
)


def build_case_report(case: CaseRecord) -> Dict[str, Any]:
    """Generate the structured Step 10 report separating confirmed findings from unverified drafts."""
    confirmed_records = [f.to_dict() for f in case.confirmed_findings]
    confirmed_hyp_ids = {f.hypothesis_id for f in case.confirmed_findings}

    unconfirmed_drafts: List[Dict[str, Any]] = []
    for hyp in case.hypotheses:
        if hyp.hypothesis_id not in confirmed_hyp_ids or hyp.verification_status == VerificationStatus.UNKNOWN:
            unconfirmed_drafts.append(
                {
                    "hypothesis_id": hyp.hypothesis_id,
                    "template_id": hyp.template_id,
                    "title": hyp.title,
                    "rendered_statement": hyp.rendered_statement,
                    "verification_status": hyp.verification_status.value,
                    "evidence_class": hyp.evidence_class.value,
                    "watermark": "DRAFT / UNCONFIRMED HYPOTHESIS - NOT A VERIFIED FAULT",
                }
            )

    prediction_section: Dict[str, Any]
    if isinstance(case.prediction, PredictionSuccess):
        p = case.prediction
        prediction_section = {
            "status": "SUCCESS",
            "evidence_class": p.evidence_class.value,
            "baseline_evidence_class": p.baseline_evidence_class.value,
            "time_constant_tau_hours": p.time_constant_tau_hours,
            "ambient_temp_c": p.ambient_temp_c,
            "r_squared": p.r_squared,
            "thermal_model_rmse_c": p.thermal_model_rmse_c,
            "naive_baseline_rmse_c": p.naive_baseline_rmse_c,
            "estimated_time_to_threshold_sec": p.estimated_time_to_threshold_sec,
            "uncertainty_95_halfwidth_c": p.uncertainty_95_halfwidth_c,
        }
    elif isinstance(case.prediction, PredictionRefusal):
        r = case.prediction
        prediction_section = {
            "status": "REFUSAL",
            "evidence_class": r.evidence_class.value,
            "reason_code": r.reason_code.value,
            "refusal_explanation": r.refusal_explanation,
            "failed_checks": r.failed_checks,
        }
    else:
        prediction_section = {"status": "NOT_RUN"}

    conditions_and_limits = [
        f"Dataset provenance is strictly '{case.evidence_class.value}'.",
        (
            f"User claimed real data: {case.user_claimed_real} "
            "(stored separately; never upgrades UNVALIDATED provenance)."
        ),
        "Thermal model is a first-order single-time-constant simplification.",
        "Unconfirmed hypotheses are template-rendered diagnostic candidates, never confirmed faults.",
    ]

    markdown_lines = [
        f"# ColdSense Incident & Diagnostic Report: {case.title}",
        f"- **Case ID:** `{case.case_id}`",
        f"- **Dataset Evidence Class:** `{case.evidence_class.value}`",
        f"- **User Claimed Real (Stored Separately):** `{case.user_claimed_real}`",
        f"- **Generated At (UTC):** `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "## 1. Technician-Confirmed Findings (Verified & Append-Only)",
    ]
    if confirmed_records:
        for rec in confirmed_records:
            markdown_lines.append(
                f"- **[{rec['verification_status']}]** Hypothesis `{rec['hypothesis_id']}` "
                f"(Check `{rec['check_id']}`) by Technician `{rec['technician_id']}` at `{rec['confirmed_at_iso']}` "
                f"| Observation: *{rec['technician_observation']}* | Hash: `{rec['record_hash'][:16]}...` "
                f"| Token: `{rec['token_id']}`"
            )
    else:
        markdown_lines.append(
            "- *No findings have been confirmed by a technician yet. All diagnostic hypotheses remain UNKNOWN.*"
        )

    markdown_lines.append("")
    markdown_lines.append("## 2. Unconfirmed Diagnostic Drafts & Open Hypotheses (Not Confirmed Faults)")
    if unconfirmed_drafts:
        for d in unconfirmed_drafts:
            markdown_lines.append(
                f"- **[DRAFT / {d['verification_status']}]** `{d['hypothesis_id']}` ({d['evidence_class']}): "
                f"{d['rendered_statement']}"
            )
    else:
        markdown_lines.append("- *All candidate hypotheses have been adjudicated by a technician.*")

    markdown_lines.append("")
    markdown_lines.append("## 3. Thermal Model & Reliability Gate Summary")
    markdown_lines.append(f"- **Prediction Outcome:** `{prediction_section['status']}`")
    if prediction_section["status"] == "SUCCESS":
        markdown_lines.append(
            f"- **Estimated Time Constant (tau):** `{prediction_section['time_constant_tau_hours']} hours` "
            f"(`R^2 = {prediction_section['r_squared']}`, Model RMSE `{prediction_section['thermal_model_rmse_c']} C` "
            f"vs Naive Baseline RMSE `{prediction_section['naive_baseline_rmse_c']} C`)"
        )
    elif prediction_section["status"] == "REFUSAL":
        markdown_lines.append(
            f"- **Refusal Reason (`{prediction_section['reason_code']}`):** {prediction_section['refusal_explanation']}"
        )

    markdown_lines.append("")
    markdown_lines.append("## 4. Epistemic Conditions & Known Limits")
    for cond in conditions_and_limits:
        markdown_lines.append(f"- {cond}")

    return {
        "case_id": case.case_id,
        "title": case.title,
        "evidence_class": case.evidence_class.value,
        "user_claimed_real": case.user_claimed_real,
        "confirmed_findings": confirmed_records,
        "unconfirmed_drafts": unconfirmed_drafts,
        "prediction_summary": prediction_section,
        "conditions_and_limits": conditions_and_limits,
        "markdown_report": "\n".join(markdown_lines),
    }
