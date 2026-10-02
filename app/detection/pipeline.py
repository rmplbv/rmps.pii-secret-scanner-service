"""Detection pipeline: runs the layers of a profile, merges, optionally asks
the LLM to adjudicate uncertain spans, and returns final spans."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from ..config import Settings
from ..models import TIMESHEET_FEATURE, Span
from .merger import Merger, normalise
from .recognizers import read_list

log = logging.getLogger(__name__)


@dataclass
class DetectionOptions:
    profile: str | None = None
    entities: list[str] | None = None          # restrict to these entity types
    ignore_entities: list[str] | None = None   # additionally ignore these
    allow_list: list[str] = field(default_factory=list)
    deny_list: list[str] = field(default_factory=list)  # always redact (entity CUSTOM)
    language: str | None = None                 # force a language instead of auto-detect
    timesheet: dict | None = None               # per-request timesheet switch overrides


@dataclass
class DetectionResult:
    spans: list[Span]
    languages: list[str]
    layers: list[str]
    timings: dict[str, float]
    llm_verdicts: int = 0
    warnings: list[str] = field(default_factory=list)


class DetectionPipeline:
    def __init__(self, settings: Settings, layers: list[str] | None = None):
        """`layers` limits which layers are loaded (e.g. a fast worker only
        loads patterns+spacy). Defaults to every layer used by any profile."""
        self.settings = settings
        wanted = set(layers or [l for p in settings.profiles.values() for l in p])
        self.loaded: dict[str, object] = {}
        from .presidio_layer import PresidioLayer

        self.presidio = PresidioLayer(settings)
        self.loaded["patterns"] = self.loaded["spacy"] = self.presidio
        if "transformer" in wanted:
            from .ml_layers import TransformerLayer
            self.loaded["transformer"] = TransformerLayer(settings)
        if "gliner" in wanted:
            from .ml_layers import GlinerLayer
            self.loaded["gliner"] = GlinerLayer(settings)
        if "llm" in wanted:
            from .llm_layer import LlmLayer
            self.loaded["llm"] = LlmLayer(settings)
        self.merger = Merger(settings)
        self.allow = {a.lower() for a in read_list(settings.path(settings.dictionaries.allow_list))}

    def supports(self, profile: str) -> bool:
        return all(l in self.loaded for l in self.settings.profiles.get(profile, []))

    def detect(self, text: str, opts: DetectionOptions | None = None) -> DetectionResult:
        opts = opts or DetectionOptions()
        s = self.settings
        profile = opts.profile or s.default_profile
        if profile not in s.profiles:
            raise ValueError(f"unknown profile {profile!r}")
        layers = s.profiles[profile]
        missing = [l for l in layers if l not in self.loaded]
        if missing:
            raise RuntimeError(f"profile {profile!r} needs layers not loaded in this worker: {missing}")

        timings: dict[str, float] = {}
        raw: list[Span] = []

        t = time.perf_counter()
        blocks = self.presidio.blocks(text) if not opts.language else None
        languages = sorted({b.language for b in blocks}) if blocks else [opts.language]
        presidio_spans = self.presidio.detect(text, opts.language)
        if "spacy" not in layers:
            presidio_spans = [x for x in presidio_spans if "spacy" not in x.sources]
        raw.extend(presidio_spans)
        timings["patterns+spacy"] = time.perf_counter() - t

        for name in ("transformer", "gliner"):
            if name in layers:
                t = time.perf_counter()
                raw.extend(self.loaded[name].detect(text))
                timings[name] = time.perf_counter() - t

        allow = self.allow | {a.lower() for a in opts.allow_list}
        ignored = self._ignored(opts)
        raw = normalise(raw, s, text, allow, ignored)
        raw.extend(self._deny_spans(text, opts.deny_list))  # explicit terms bypass all filters

        verdict_count = 0
        warnings: list[str] = []
        if "llm" in layers:
            t = time.perf_counter()
            raw, verdict_count = self._llm_pass(text, raw, allow, ignored, warnings)
            timings["llm"] = time.perf_counter() - t

        spans = self.merger.merge(raw, text)
        if opts.entities:
            spans = [x for x in spans if x.entity in set(opts.entities)]
        return DetectionResult(spans, languages, layers, {k: round(v, 3) for k, v in timings.items()},
                               verdict_count, warnings)

    def _ignored(self, opts: DetectionOptions) -> set[str]:
        ignored = set(self.settings.ignored_entities) | set(opts.ignore_entities or [])
        ts = self.settings.timesheet.merged(opts.timesheet)
        for entity, feature in TIMESHEET_FEATURE.items():
            if not getattr(ts, feature).detect:
                ignored.add(entity)
        return ignored

    def _deny_spans(self, text: str, terms: list[str]) -> list[Span]:
        if not terms:
            return []
        from .recognizers import _term_regex

        rx = _term_regex(terms)
        return [Span(m.start(), m.end(), "CUSTOM", 1.0, {"patterns"}, m.group()) for m in rx.finditer(text)]

    def _llm_pass(self, text: str, raw: list[Span], allow: set[str], ignored: set[str],
                  warnings: list[str]) -> tuple[list[Span], int]:
        lo, hi = self.settings.layers.llm.adjudicate_range
        prelim = self.merger.merge(raw, text, apply_threshold=False)
        uncertain = [x for x in prelim if lo <= x.score < hi]
        known = [x for x in prelim if x.score >= hi]
        found, verdicts = self.loaded["llm"].detect(text, uncertain, known, warnings)
        found = normalise(found, self.settings, text, allow, ignored)
        for v in verdicts:
            c = v.span
            if v.is_pii:
                found.append(Span(c.start, c.end, v.entity if v.entity and v.entity != "OTHER_PII" else c.entity,
                                  0.85, {"llm"}, c.text))
            else:
                # LLM says "not PII": damp every raw span behind this candidate
                for r in raw:
                    if r.entity == c.entity and r.overlaps(c):
                        r.score *= 0.5
        return raw + found, len(verdicts)
