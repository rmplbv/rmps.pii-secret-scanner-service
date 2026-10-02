"""Model-holding core used by the worker: detection + anonymization for one
request. Free of Redis/HTTP so it can be tested directly."""

from __future__ import annotations

import logging
import time
from collections import Counter

from .anonymize.engine import Anonymizer
from .anonymize.session import SessionState
from .api.schemas import AnalyzeRequest, AnonymizeRequest
from .config import Settings
from .detection.pipeline import DetectionOptions, DetectionPipeline

log = logging.getLogger(__name__)


class PiiService:
    def __init__(self, settings: Settings, layers: list[str] | None = None):
        self.settings = settings
        self.pipeline = DetectionPipeline(settings, layers)
        self.anonymizer = Anonymizer(settings)

    def _detect(self, req: AnalyzeRequest | AnonymizeRequest):
        opts = DetectionOptions(
            profile=req.profile, entities=req.entities, ignore_entities=req.ignore_entities,
            allow_list=req.allow_list, deny_list=req.deny_list, language=req.language,
            timesheet=req.timesheet.as_dict() if req.timesheet else None,
        )
        return self.pipeline.detect(req.text, opts)

    def _blocks(self, text: str, language: str | None):
        if language:
            return [(0, len(text), language)]
        return [(b.start, b.end, b.language) for b in self.pipeline.presidio.blocks(text)]

    def analyze(self, req: AnalyzeRequest) -> dict:
        t = time.perf_counter()
        det = self._detect(req)
        return {
            "entities": [s.to_dict(include_text=req.include_text) for s in det.spans],
            "stats": dict(Counter(s.entity for s in det.spans)),
            "languages": det.languages,
            "profile": req.profile or self.settings.default_profile,
            "layers": det.layers,
            "timings": {**det.timings, "total": round(time.perf_counter() - t, 3)},
            "warnings": det.warnings,
        }

    def anonymize(self, req: AnonymizeRequest, state: SessionState | None = None) -> tuple[dict, SessionState]:
        t = time.perf_counter()
        state = state or SessionState()
        det = self._detect(req)
        ts = self.settings.timesheet.merged(req.timesheet.as_dict() if req.timesheet else None)
        text, reps = self.anonymizer.anonymize(
            req.text, det.spans, state, req.mode, req.operators, ts, self._blocks(req.text, req.language))
        result = {
            "text": text,
            "session_id": state.session_id if (req.reversible or req.session_id) else None,
            "reversible": bool(req.reversible or req.session_id),
            "entities": [r.to_dict(include_original=req.include_originals) for r in reps],
            "stats": dict(Counter(s.entity for s in det.spans)),
            "languages": det.languages,
            "profile": req.profile or self.settings.default_profile,
            "layers": det.layers,
            "timings": {**det.timings, "total": round(time.perf_counter() - t, 3)},
            "warnings": det.warnings,
        }
        return result, state
