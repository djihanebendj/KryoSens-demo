"""Step 8 (Technician Finding Review) & Step 9 (Trusted Confirmation & Append-Only Store).

Enforces:
- A technician's confirmation uses a single-use, expiring cryptographic token bound to the
  SHA-256 hash of the exact FindingDraftRecord.
- The language model / Copilot cannot confirm anything.
- Until explicitly confirmed or rejected by a technician, every hypothesis outcome stays UNKNOWN.
- Confirmed findings are stored append-only (immutable) with their source, token_id, and timestamp.
"""

from __future__ import annotations

import hashlib
import hmac

from coldsense.app.security import resolve_secret
import secrets
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from coldsense.models import (
    CaseRecord,
    EvidenceClass,
    FindingDraftRecord,
    FindingRecord,
    Hypothesis,
    TrustedConfirmationToken,
    VerificationStatus,
)


class ConfirmationError(Exception):
    """Raised when a trusted confirmation token is invalid, expired, replayed, or mismatched."""


class TrustedConfirmationManager:
    """Manages single-use, hash-bound confirmation tokens and an append-only SQLite/memory audit log."""

    def __init__(
        self,
        secret_key: Optional[str] = None,
        token_ttl_sec: float = 300.0,
        sqlite_path: str = ":memory:",
    ) -> None:
        self._secret = resolve_secret("COLDSENSE_CONFIRMATION_SECRET", secret_key, "confirmation token")
        self._token_ttl_sec = token_ttl_sec
        self._tokens: Dict[str, TrustedConfirmationToken] = {}
        self._conn = sqlite3.connect(sqlite_path, check_same_thread=False)
        self._init_append_only_table()

    def _init_append_only_table(self) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS findings_audit_log (
                finding_id TEXT PRIMARY KEY,
                case_id TEXT NOT NULL,
                hypothesis_id TEXT NOT NULL,
                check_id TEXT NOT NULL,
                verification_status TEXT NOT NULL,
                technician_id TEXT NOT NULL,
                technician_observation TEXT NOT NULL,
                confirmed_at_iso TEXT NOT NULL,
                confirmed_at_epoch REAL NOT NULL,
                record_hash TEXT NOT NULL,
                token_id TEXT NOT NULL UNIQUE,
                source_evidence_class TEXT NOT NULL
            )
            """
        )
        # SQLite triggers enforcing strict append-only immutability (no UPDATE or DELETE allowed)
        cur.execute(
            """
            CREATE TRIGGER IF NOT EXISTS prevent_findings_update
            BEFORE UPDATE ON findings_audit_log
            BEGIN
                SELECT RAISE(ABORT, 'Append-only violation: findings_audit_log records cannot be updated.');
            END;
            """
        )
        cur.execute(
            """
            CREATE TRIGGER IF NOT EXISTS prevent_findings_delete
            BEFORE DELETE ON findings_audit_log
            BEGIN
                SELECT RAISE(ABORT, 'Append-only violation: findings_audit_log records cannot be deleted.');
            END;
            """
        )
        self._conn.commit()

    def _compute_signature(
        self, token_id: str, case_id: str, record_hash: str, expires_at: float
    ) -> str:
        msg = f"{token_id}:{case_id}:{record_hash}:{expires_at:.3f}".encode("utf-8")
        return hmac.new(self._secret, msg, hashlib.sha256).hexdigest()

    def stage_draft_finding(
        self,
        case: CaseRecord,
        hypothesis_id: str,
        check_id: str,
        proposed_status: VerificationStatus,
        technician_id: str,
        technician_observation: str,
        now_epoch: Optional[float] = None,
    ) -> Tuple[FindingDraftRecord, TrustedConfirmationToken]:
        """Step 8: Stage a draft finding for technician review and mint a single-use token
        bound to the SHA-256 hash of the exact record.
        """
        if proposed_status not in (VerificationStatus.CONFIRMED, VerificationStatus.REJECTED):
            raise ConfirmationError(
                "Technician must state a real verification outcome (CONFIRMED or REJECTED)."
            )
        if not technician_id or not technician_id.strip():
            raise ConfirmationError("A valid human technician_id is required.")
        if technician_id.lower() in ("llm", "copilot", "ai", "model", "assistant", "system"):
            raise ConfirmationError("Rule violation: The language model or system cannot confirm findings.")
        if not technician_observation or len(technician_observation.strip()) < 5:
            raise ConfirmationError("Technician must provide a concrete physical observation note.")

        target_hyp = next((h for h in case.hypotheses if h.hypothesis_id == hypothesis_id), None)
        if target_hyp is None:
            raise ConfirmationError(f"Unknown hypothesis_id: {hypothesis_id}")

        draft = FindingDraftRecord(
            case_id=case.case_id,
            hypothesis_id=hypothesis_id,
            check_id=check_id,
            proposed_status=proposed_status,
            technician_id=technician_id.strip(),
            technician_observation=technician_observation.strip(),
            evidence_class=case.evidence_class,
        )
        rec_hash = draft.record_hash()
        now = now_epoch if now_epoch is not None else time.time()
        exp = now + self._token_ttl_sec
        token_id = f"tct-{secrets.token_hex(12)}"
        sig = self._compute_signature(token_id, case.case_id, rec_hash, exp)

        token = TrustedConfirmationToken(
            token_id=token_id,
            case_id=case.case_id,
            record_hash=rec_hash,
            issued_at_epoch=now,
            expires_at_epoch=exp,
            signature=sig,
            used=False,
        )
        self._tokens[token_id] = token
        case.draft_findings[rec_hash] = draft
        return draft, token

    def confirm_finding(
        self,
        case: CaseRecord,
        draft: FindingDraftRecord,
        token_id: str,
        signature: str,
        now_epoch: Optional[float] = None,
        caller_role: str = "technician",
    ) -> FindingRecord:
        """Step 9: Verify the single-use token, expiry, and exact record hash, then commit
        an immutable FindingRecord to the append-only store.
        """
        if caller_role.lower() != "technician":
            raise ConfirmationError("Only a human technician can execute confirmation.")

        stored_token = self._tokens.get(token_id)
        if stored_token is None:
            raise ConfirmationError("Invalid or unknown confirmation token.")
        if stored_token.used:
            raise ConfirmationError("Replay blocked: confirmation token has already been used.")

        now = now_epoch if now_epoch is not None else time.time()
        if now > stored_token.expires_at_epoch:
            raise ConfirmationError("Confirmation token has expired.")

        expected_sig = self._compute_signature(
            stored_token.token_id,
            stored_token.case_id,
            stored_token.record_hash,
            stored_token.expires_at_epoch,
        )
        if not hmac.compare_digest(expected_sig, signature) or not hmac.compare_digest(
            stored_token.signature, signature
        ):
            raise ConfirmationError("Invalid confirmation token cryptographic signature.")

        computed_hash = draft.record_hash()
        if not hmac.compare_digest(computed_hash, stored_token.record_hash):
            raise ConfirmationError(
                "Record hash mismatch: submitted record differs from the token-bound record."
            )
        if draft.case_id != case.case_id:
            raise ConfirmationError("Case ID mismatch on confirmation token.")

        # Mark token used (single-use enforcement)
        self._tokens[token_id] = TrustedConfirmationToken(
            token_id=stored_token.token_id,
            case_id=stored_token.case_id,
            record_hash=stored_token.record_hash,
            issued_at_epoch=stored_token.issued_at_epoch,
            expires_at_epoch=stored_token.expires_at_epoch,
            signature=stored_token.signature,
            used=True,
        )

        iso_now = datetime.fromtimestamp(now, tz=timezone.utc).isoformat()
        finding = FindingRecord(
            finding_id=f"fnd-{uuid.uuid4().hex[:12]}",
            case_id=case.case_id,
            hypothesis_id=draft.hypothesis_id,
            check_id=draft.check_id,
            verification_status=draft.proposed_status,
            technician_id=draft.technician_id,
            technician_observation=draft.technician_observation,
            confirmed_at_iso=iso_now,
            confirmed_at_epoch=now,
            record_hash=computed_hash,
            token_id=token_id,
            source_evidence_class=draft.evidence_class,
        )

        # Persist to append-only SQLite audit log
        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT INTO findings_audit_log (
                finding_id, case_id, hypothesis_id, check_id, verification_status,
                technician_id, technician_observation, confirmed_at_iso, confirmed_at_epoch,
                record_hash, token_id, source_evidence_class
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                finding.finding_id,
                finding.case_id,
                finding.hypothesis_id,
                finding.check_id,
                finding.verification_status.value,
                finding.technician_id,
                finding.technician_observation,
                finding.confirmed_at_iso,
                finding.confirmed_at_epoch,
                finding.record_hash,
                finding.token_id,
                finding.source_evidence_class.value,
            ),
        )
        self._conn.commit()

        # Update case append-only list and reflect status on the matching hypothesis
        case.confirmed_findings.append(finding)
        updated_hyps: List[Hypothesis] = []
        for h in case.hypotheses:
            if h.hypothesis_id == draft.hypothesis_id:
                updated_hyps.append(
                    Hypothesis(
                        hypothesis_id=h.hypothesis_id,
                        template_id=h.template_id,
                        title=h.title,
                        rendered_statement=h.rendered_statement,
                        evidence_class=h.evidence_class,
                        supporting_evidence_ids=h.supporting_evidence_ids,
                        contradicting_evidence_ids=h.contradicting_evidence_ids,
                        prior_plausibility_rank=h.prior_plausibility_rank,
                        verification_status=draft.proposed_status,
                    )
                )
            else:
                updated_hyps.append(h)
        case.hypotheses = updated_hyps
        return finding

    def list_audit_log(self, case_id: Optional[str] = None) -> List[FindingRecord]:
        cur = self._conn.cursor()
        if case_id:
            cur.execute(
                "SELECT * FROM findings_audit_log WHERE case_id = ? ORDER BY confirmed_at_epoch ASC",
                (case_id,),
            )
        else:
            cur.execute("SELECT * FROM findings_audit_log ORDER BY confirmed_at_epoch ASC")
        rows = cur.fetchall()
        return [
            FindingRecord(
                finding_id=r[0],
                case_id=r[1],
                hypothesis_id=r[2],
                check_id=r[3],
                verification_status=VerificationStatus(r[4]),
                technician_id=r[5],
                technician_observation=r[6],
                confirmed_at_iso=r[7],
                confirmed_at_epoch=float(r[8]),
                record_hash=r[9],
                token_id=r[10],
                source_evidence_class=EvidenceClass(r[11]),
            )
            for r in rows
        ]
