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
from .formats import html as html_fmt

log = logging.getLogger(__name__)


class PiiService:
    def __init__(self, settings: Settings, layers: list[str] | None = None):
        self.settings = settings
        self.pipeline = DetectionPipeline(settings, layers)
        self.anonymizer = Anonymizer(settings)

    def _prepare(self, req: AnalyzeRequest | AnonymizeRequest):
        """Returns (analysed text, html virtual text or None)."""
        fmt = req.format
        if fmt == "auto":
            fmt = "html" if html_fmt.looks_like_html(req.text) else "text"
        if fmt == "html":
            h = self.settings.html
            vt = html_fmt.extract(req.text, h.attributes, h.skip_elements)
            return vt.text, vt
        return req.text, None

    def _detect(self, req: AnalyzeRequest | AnonymizeRequest, text: str):
        opts = DetectionOptions(
            profile=req.profile, entities=req.entities, ignore_entities=req.ignore_entities,
            allow_list=req.allow_list, deny_list=req.deny_list, language=req.language,
            timesheet=req.timesheet.as_dict() if req.timesheet else None, preset=req.preset,
        )
        return self.pipeline.detect(text, opts)

    def _blocks(self, text: str, language: str | None):
        if language:
            return [(0, len(text), language)]
        return [(b.start, b.end, b.language) for b in self.pipeline.presidio.blocks(text)]

    @staticmethod
    def _to_raw(d: dict, vt) -> dict:
        if vt is not None:
            raw = html_fmt.to_raw_span(vt, d["start"], d["end"])
            if raw:
                d["start"], d["end"] = raw
            d.pop("new_start", None)
            d.pop("new_end", None)
        return d

    def _meta(self, req, det, t0: float, vt) -> dict:
        return {
            "stats": dict(Counter(s.entity for s in det.spans)),
            "languages": det.languages,
            "profile": req.profile or self.settings.default_profile,
            "preset": det.preset,
            "format": "html" if vt is not None else "text",
            "layers": det.layers,
            "timings": {**det.timings, "total": round(time.perf_counter() - t0, 3)},
            "warnings": det.warnings,
        }

    def analyze(self, req: AnalyzeRequest) -> dict:
        t = time.perf_counter()
        text, vt = self._prepare(req)
        det = self._detect(req, text)
        return {"entities": [self._to_raw(s.to_dict(include_text=req.include_text), vt) for s in det.spans],
                **self._meta(req, det, t, vt)}

    def anonymize(self, req: AnonymizeRequest, state: SessionState | None = None) -> tuple[dict, SessionState]:
        t = time.perf_counter()
        state = state or SessionState()
        text, vt = self._prepare(req)
        det = self._detect(req, text)
        out, reps = self.anonymizer.anonymize(
            text, det.spans, state, req.mode, req.operators, det.timesheet, self._blocks(text, req.language))
        if vt is not None:
            out = html_fmt.apply(req.text, vt, [(r.span.start, r.span.end, r.replacement) for r in reps])
        result = {
            "text": out,
            "session_id": state.session_id if (req.reversible or req.session_id) else None,
            "reversible": bool(req.reversible or req.session_id),
            "entities": [self._to_raw(r.to_dict(include_original=req.include_originals), vt) for r in reps],
            **self._meta(req, det, t, vt),
        }
        return result, state
