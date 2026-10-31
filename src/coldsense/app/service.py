"""Typed Backend Service Layer orchestrating all 10 ColdSense workflow steps and 11 app views."""

from __future__ import annotations

import os

from typing import Any, Dict, List, Optional, Tuple

from coldsense.app.case_store import CaseStore, RateLimiter, RedactedLogger
from coldsense.app.security import SessionSecurityManager
from coldsense.confirmation import TrustedConfirmationManager
from coldsense.copilot import TechnicianCopilot
from coldsense.diagnostics import build_evidence_board, generate_diagnostics_and_next_checks
from coldsense.events import detect_temperature_events
from coldsense.ingestion import (
    create_uploaded_case,
    evaluate_data_quality,
    generate_simulated_demo_case,
)
from coldsense.knowledge_base import LocalKnowledgeBase
from coldsense.models import (
    FindingDraftRecord,
    PredictionSuccess,
    VerificationStatus,
    WorkflowStep,
)
from coldsense.reports import build_case_report
from coldsense.thermal import run_thermal_prediction


class WorkflowGateError(Exception):
    """Raised when a user tries to access a step before prerequisite steps are completed."""


def copilot_enabled_from_env() -> bool:
    """The only switch for the copilot: the server environment. Default OFF.

    No HTTP route and no browser action can change it; a deployment that wants the copilot sets
    COLDSENSE_COPILOT_ENABLED=true (and supplies a key) before the process starts.
    """
    return os.environ.get("COLDSENSE_COPILOT_ENABLED", "false").strip().lower() in ("1", "true", "yes", "on")


class ColdSenseService:
    """Unified application service backing the 11 ColdSense views and 10 workflow steps."""

    def __init__(
        self,
        copilot_enabled: bool = False,
        semantic_search_available: bool = True,
        case_ttl_sec: float = 3600.0,
        rate_limit_per_min: int = 120,
    ) -> None:
        self.case_store = CaseStore(default_ttl_sec=case_ttl_sec)
        self.security = SessionSecurityManager()
        self.rate_limiter = RateLimiter(max_requests=rate_limit_per_min, window_sec=60.0)
        self.logger = RedactedLogger()
        # The Gemini network path is opened only when the copilot is enabled by the server AND a key is
        # present. Without this the orchestrator was always constructed with allow_network=False, so the
        # engine could never be GEMINI however valid the key was.
        gemini_key = (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip() or None
        self.kb = LocalKnowledgeBase(semantic_available=semantic_search_available)
        self.copilot = TechnicianCopilot(
            knowledge_base=self.kb,
            enabled=copilot_enabled,
            allow_network_gemini=bool(copilot_enabled and gemini_key),
            gemini_api_key=gemini_key,
        )
        self.confirmation_mgr = TrustedConfirmationManager()

    def get_dashboard(self, session_id: str) -> Dict[str, Any]:
        cases = self.case_store.list_for_session(session_id)
        return {
            "page": "Dashboard",
            "active_cases": [c.to_summary_dict() for c in cases],
            "copilot_feature_enabled": self.copilot.enabled,
            "retrieval_mode": (
                "SEMANTIC_HYBRID" if self.kb.semantic_available else "LEXICAL_FALLBACK"
            ),
            "workflow_steps": [
                {"step": s.value, "name": s.name} for s in WorkflowStep
            ],
        }

    # --- Step 1 / View 2: Upload Data & Demo Case ---
    def upload_case(
        self,
        session_id: str,
        raw_content: str,
        fmt: str = "csv",
        title: str = "Uploaded Cold-Room Log",
        user_claimed_real: bool = False,
        critical_threshold_c: float = 8.0,
    ) -> Dict[str, Any]:
        self.rate_limiter.check(session_id)
        case, total_rows, missing_rows = create_uploaded_case(
            owner_session_id=session_id,
            raw_content=raw_content,
            fmt=fmt,
            title=title,
            user_claimed_real=user_claimed_real,
            critical_threshold_c=critical_threshold_c,
            ttl_seconds=self.case_store.default_ttl_sec,
        )
        # Store raw parsing counters for Step 2
        case._raw_total_rows = total_rows  # type: ignore[attr-defined]
        case._raw_missing_rows = missing_rows  # type: ignore[attr-defined]
        self.case_store.put(case)
        self.logger.log(
            "CASE_UPLOAD",
            f"Session {session_id} uploaded case {case.case_id} ({case.evidence_class.value})",
        )
        return case.to_summary_dict()

    def create_demo_case(
        self,
        session_id: str,
        scenario: str = "power_cut",
    ) -> Dict[str, Any]:
        self.rate_limiter.check(session_id)
        case = generate_simulated_demo_case(
            owner_session_id=session_id,
            scenario=scenario,
            ttl_seconds=self.case_store.default_ttl_sec,
        )
        self.case_store.put(case)
        self.logger.log(
            "DEMO_CASE_CREATED",
            f"Session {session_id} created SIMULATED case {case.case_id} ({scenario})",
        )
        return case.to_summary_dict()

    # --- Step 2 / View 3: Data Quality ---
    def run_step2_data_quality(self, session_id: str, case_id: str) -> Dict[str, Any]:
        self.rate_limiter.check(session_id)
        case = self.case_store.get(case_id, session_id)
        total_rows = getattr(case, "_raw_total_rows", None)
        missing_rows = getattr(case, "_raw_missing_rows", 0)
        report = evaluate_data_quality(
            case, total_raw_rows=total_rows, missing_value_rows=missing_rows
        )
        return {
            "page": "Data Quality",
            "unlocked_step": case.unlocked_step,
            "data_quality": report.to_dict(),
        }

    # --- Step 3 / View 4: Temperature Timeline & Event Analysis ---
    def run_step3_timeline_events(self, session_id: str, case_id: str) -> Dict[str, Any]:
        self.rate_limiter.check(session_id)
        case = self.case_store.get(case_id, session_id)
        if not case.is_step_unlocked(WorkflowStep.EVENT_ANALYSIS):
            raise WorkflowGateError(
                "Step 3 (Temperature Timeline) is locked until Step 2 (Data Quality) passes."
            )
        events = detect_temperature_events(case)
        return {
            "page": "Temperature Timeline",
            "case_id": case.case_id,
            "evidence_class": case.evidence_class.value,
            "unlocked_step": case.unlocked_step,
            "points": [p.to_dict() for p in case.points],
            "events": [e.to_dict() for e in events],
        }

    # --- Step 4 / View 5: Prediction (First-Order Model vs Naive Baseline or Refusal) ---
    def run_step4_prediction(
        self,
        session_id: str,
        case_id: str,
        event_id: Optional[str] = None,
        horizon_sec: float = 3600.0,
    ) -> Dict[str, Any]:
        self.rate_limiter.check(session_id)
        case = self.case_store.get(case_id, session_id)
        if not case.is_step_unlocked(WorkflowStep.PREDICTION):
            raise WorkflowGateError(
                "Step 4 (Prediction) is locked until Step 3 (Event Analysis) completes."
            )
        outcome = run_thermal_prediction(case, event_id=event_id, horizon_sec=horizon_sec)
        return {
            "page": "Prediction",
            "unlocked_step": case.unlocked_step,
            "prediction": outcome.to_dict(),
        }

    # --- Step 5 / View 6: Evidence Board ---
    def run_step5_evidence(self, session_id: str, case_id: str) -> Dict[str, Any]:
        self.rate_limiter.check(session_id)
        case = self.case_store.get(case_id, session_id)
        if not case.is_step_unlocked(WorkflowStep.EVIDENCE):
            raise WorkflowGateError(
                "Step 5 (Evidence) is locked until Step 4 (Prediction) completes."
            )
        board = build_evidence_board(case)
        return {
            "page": "Evidence",
            "unlocked_step": case.unlocked_step,
            "evidence_board": board.to_dict(),
        }

    # --- Step 6 / View 7: Next Best Check & Hypotheses ---
    def run_step6_next_best_check(self, session_id: str, case_id: str) -> Dict[str, Any]:
        self.rate_limiter.check(session_id)
        case = self.case_store.get(case_id, session_id)
        if not case.is_step_unlocked(WorkflowStep.NEXT_BEST_CHECK):
            raise WorkflowGateError(
                "Step 6 (Next Best Check) is locked until Step 5 (Evidence) completes."
            )
        hyps, checks = generate_diagnostics_and_next_checks(case)
        return {
            "page": "Next Best Check",
            "case_id": case.case_id,
            "evidence_class": case.evidence_class.value,
            "unlocked_step": case.unlocked_step,
            "hypotheses": [h.to_dict() for h in hyps],
            "next_best_checks": [c.to_dict() for c in checks],
        }

    # --- Step 7 / View 8: Technician Copilot ---
    def run_step7_copilot(
        self,
        session_id: str,
        query: str,
        case_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        self.rate_limiter.check(session_id)
        case = self.case_store.get(case_id, session_id) if case_id else None
        result = self.copilot.ask(query=query, case=case)
        engine, notice = self._engine_report(result)
        self.logger.log(
            "COPILOT_QUERY",
            f"Session {session_id} copilot response_type={result.response_type} engine={engine}",
        )
        return {
            "page": "Technician Copilot",
            "copilot_enabled": self.copilot.enabled,
            "engine": engine,
            "engine_notice": notice,
            "result": result.to_dict(),
        }

    def _engine_report(self, result: Any) -> Tuple[str, Optional[str]]:
        """Which engine produced the answer, and a short visible notice when Gemini did not.

        A hazard template and a disabled copilot are produced by rules, not by a model, so they are
        reported as RULE_TEMPLATE rather than as a fallback.
        """
        if getattr(result, "response_type", "") == "HAZARD_TEMPLATE":
            return "RULE_TEMPLATE", None
        if not self.copilot.enabled:
            return "NONE", "The copilot is disabled on this server; no model was called."
        llm = getattr(self.copilot, "gemini_llm", None)
        raw = (getattr(llm, "last_engine_used", "") or "").upper()
        failure = getattr(llm, "last_engine_error", None)
        if "GEMINI" in raw:
            return "GEMINI", None
        notice = "Answer produced without Gemini (deterministic extractive fallback)."
        if failure:
            notice += f" Reason: {failure}."
        return "EXTRACTIVE_FALLBACK", notice

    # --- Step 8 & 9 / View 9: Investigation (Stage Finding & Trusted Confirmation) ---
    def get_investigation_view(self, session_id: str, case_id: str) -> Dict[str, Any]:
        case = self.case_store.get(case_id, session_id)
        if not case.is_step_unlocked(WorkflowStep.TECHNICIAN_FINDING):
            raise WorkflowGateError(
                "Investigation view is locked until Step 6 (Next Best Check) completes."
            )
        return {
            "page": "Investigation",
            "case_id": case.case_id,
            "evidence_class": case.evidence_class.value,
            "hypotheses": [h.to_dict() for h in case.hypotheses],
            "next_best_checks": [c.to_dict() for c in case.next_best_checks],
            "confirmed_findings": [f.to_dict() for f in case.confirmed_findings],
        }

    def stage_finding_for_confirmation(
        self,
        session_id: str,
        case_id: str,
        hypothesis_id: str,
        check_id: str,
        proposed_status: str,
        technician_id: str,
        technician_observation: str,
    ) -> Dict[str, Any]:
        self.rate_limiter.check(session_id)
        case = self.case_store.get(case_id, session_id)
        if not case.is_step_unlocked(WorkflowStep.TECHNICIAN_FINDING):
            raise WorkflowGateError(
                "Step 8 (Technician Finding) requires Step 6 (Next Best Check) to be completed first."
            )
        status_enum = VerificationStatus(proposed_status)
        draft, token = self.confirmation_mgr.stage_draft_finding(
            case=case,
            hypothesis_id=hypothesis_id,
            check_id=check_id,
            proposed_status=status_enum,
            technician_id=technician_id,
            technician_observation=technician_observation,
        )
        self.logger.log(
            "STAGE_FINDING",
            f"Session {session_id} minted token {token.token_id} for hash {draft.record_hash()}",
        )
        return {
            "step": 8,
            "draft_record": draft.to_dict(),
            "trusted_confirmation_token": token.to_dict(),
        }

    def confirm_staged_finding(
        self,
        session_id: str,
        case_id: str,
        hypothesis_id: str,
        check_id: str,
        proposed_status: str,
        technician_id: str,
        technician_observation: str,
        token_id: str,
        signature: str,
        caller_role: str = "technician",
    ) -> Dict[str, Any]:
        self.rate_limiter.check(session_id)
        case = self.case_store.get(case_id, session_id)
        if not case.is_step_unlocked(WorkflowStep.VERIFICATION):
            raise WorkflowGateError(
                "Step 9 (Verification) is locked until Step 6 completes."
            )
        draft = FindingDraftRecord(
            case_id=case.case_id,
            hypothesis_id=hypothesis_id,
            check_id=check_id,
            proposed_status=VerificationStatus(proposed_status),
            technician_id=technician_id,
            technician_observation=technician_observation,
            evidence_class=case.evidence_class,
        )
        finding = self.confirmation_mgr.confirm_finding(
            case=case,
            draft=draft,
            token_id=token_id,
            signature=signature,
            caller_role=caller_role,
        )
        self.logger.log(
            "CONFIRM_FINDING",
            f"Session {session_id} confirmed finding {finding.finding_id} using token {token_id}",
        )
        return {
            "step": 9,
            "confirmed_finding": finding.to_dict(),
        }

    # --- View 10: Incident Overview ---
    def get_incident_view(self, session_id: str, case_id: str) -> Dict[str, Any]:
        case = self.case_store.get(case_id, session_id)
        if not case.is_step_unlocked(WorkflowStep.NEXT_BEST_CHECK):
            raise WorkflowGateError("Incident view requires event & diagnostic analysis first.")
        peak_c = max((p.temp_c for p in case.points), default=0.0)
        breached = peak_c >= case.critical_threshold_c
        return {
            "page": "Incident",
            "case_id": case.case_id,
            "title": case.title,
            "evidence_class": case.evidence_class.value,
            "critical_threshold_c": case.critical_threshold_c,
            "peak_observed_temp_c": round(peak_c, 3),
            "threshold_breached": breached,
            "events": [e.to_dict() for e in case.events],
            "prediction_type": (
                "SUCCESS" if isinstance(case.prediction, PredictionSuccess) else "REFUSAL"
            ),
            "confirmed_findings_count": len(case.confirmed_findings),
        }

    # --- Step 10 / View 11: Reports ---
    def get_step10_report(self, session_id: str, case_id: str) -> Dict[str, Any]:
        self.rate_limiter.check(session_id)
        case = self.case_store.get(case_id, session_id)
        if not case.is_step_unlocked(WorkflowStep.REPORT):
            raise WorkflowGateError(
                "Step 10 (Report) is locked until Step 6 (Next Best Check) completes."
            )
        report = build_case_report(case)
        return {
            "page": "Reports",
            "report": report,
        }
