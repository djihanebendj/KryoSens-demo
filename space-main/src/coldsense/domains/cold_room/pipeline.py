"""One call from file to answer: ingest -> validate -> segment -> predict."""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .baseline import predict_baseline
from .events import segment_events
from .ingest import IngestOptions, read_csv, read_dataframe
from .schemas import (
    DataQualityReport,
    EvidenceClass,
    Prediction,
    SegmentationParams,
    SegmentationResult,
    ValidationLimits,
    Verdict,
    WarmingSegment,
)
from .segments import detect_warming_segments
from .validate import validate

log = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    data: pd.DataFrame          # usable rows only: the analysis input
    report: DataQualityReport
    segments: list[WarmingSegment]
    observations: pd.DataFrame | None = None  # every input row with raw text, status and flags

    def predict(self, threshold_c: float, current_temp_c: float | None = None) -> Prediction:
        t0 = float(self.data["temp_c"].iloc[-1]) if current_temp_c is None else current_temp_c
        pred = predict_baseline(self.report, self.segments, t0, threshold_c)
        log.info("prediction method=%s evidence=%s refused=%s minutes=%s reasons=%d",
                 pred.method, pred.evidence.value, pred.refused, pred.minutes_to_threshold,
                 len(pred.reasons))
        return pred


def validate_source(src: str | Path | io.IOBase | pd.DataFrame, *, source: str | None = None,
                    evidence: EvidenceClass = EvidenceClass.UNVALIDATED,
                    options: IngestOptions | None = None, limits: ValidationLimits | None = None,
                    ) -> tuple[pd.DataFrame, pd.DataFrame, DataQualityReport]:
    """Ingest + validate only. The input (file or DataFrame) is never modified.

    DataFrame input is read cell by cell (no CSV round-trip); file-only options
    are rejected. See ingest.read_dataframe for the contract."""
    declared_simulated = False
    if isinstance(src, pd.DataFrame):
        declared_simulated = src.attrs.get("coldsense_evidence") == "SIMULATED"
        source = source or "<DataFrame>"
        ingested = read_dataframe(src, options, source=source)
    else:
        ingested = read_csv(src, options)
    clean, obs, report = validate(ingested, source=source, evidence=evidence, limits=limits,
                                  declared_simulated=declared_simulated)
    log.info("validation source=%s evidence=%s verdict=%s rows=%d usable=%d issues=%s",
             report.source, report.evidence.value, report.verdict.value, report.n_rows_raw,
             report.n_rows_clean, [i.code for i in report.issues])
    return clean, obs, report


def analyze(src: str | Path | io.IOBase | pd.DataFrame, *, source: str | None = None,
            evidence: EvidenceClass = EvidenceClass.UNVALIDATED,
            options: IngestOptions | None = None, limits: ValidationLimits | None = None) -> AnalysisResult:
    clean, obs, report = validate_source(src, source=source, evidence=evidence, options=options, limits=limits)
    segs = [] if report.verdict == Verdict.REFUSED else detect_warming_segments(clean, report.median_interval_s)
    log.info("analysis source=%s segments=%d clean=%d", report.source, len(segs), sum(s.clean for s in segs))
    return AnalysisResult(clean, report, segs, obs)


def segment_source(src: str | Path | io.IOBase | pd.DataFrame, *, source: str | None = None,
                   evidence: EvidenceClass = EvidenceClass.UNVALIDATED,
                   options: IngestOptions | None = None, limits: ValidationLimits | None = None,
                   params: SegmentationParams | None = None,
                   ) -> tuple[pd.DataFrame, pd.DataFrame, DataQualityReport, SegmentationResult]:
    """Ingest + validate + event segmentation (Phase 3). No prediction."""
    clean, obs, report = validate_source(src, source=source, evidence=evidence, options=options, limits=limits)
    seg = segment_events(clean, report, obs, params)
    log.info("segmentation source=%s refused=%s events=%d", report.source, seg.refused, len(seg.events))
    return clean, obs, report, seg
