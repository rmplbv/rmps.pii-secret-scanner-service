"""Detection pipeline: runs the layers of a profile, merges, optionally asks
the LLM to adjudicate uncertain spans, and returns final spans."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from ..config import Settings, Timesheet
from ..models import TIMESHEET_FEATURE, Span
from .merger import Merger, normalise
from .presets import resolve_preset
from .recognizers import read_list
from .tables import propagate_columns

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
    preset: str | None = None                   # auto | general | timesheet


@dataclass
class DetectionResult:
    spans: list[Span]
    languages: list[str]
    layers: list[str]
    timings: dict[str, float]
    llm_verdicts: int = 0
    warnings: list[str] = field(default_factory=list)
    preset: str = "general"
    timesheet: Timesheet | None = None


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
        species = read_list(settings.path(settings.dictionaries.species))
        # "~Fink": also a surname -> only suppress non-PERSON detections
        self.species = ({x.lower() for x in species if not x.startswith("~")},
                        {x[1:].strip().lower() for x in species if x.startswith("~")})

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
        raw = self.collect(text, [l for l in layers if l != "llm"], opts.language, timings)
        return self.finalize(text, raw, opts, layers, timings, use_llm="llm" in layers)

    def collect(self, text: str, layers: list[str], language: str | None = None,
                timings: dict[str, float] | None = None) -> dict[str, list[Span]]:
        """Raw (un-merged) spans per layer. "patterns" and "spacy" share one run."""
        timings = timings if timings is not None else {}
        out: dict[str, list[Span]] = {}
        if "patterns" in layers or "spacy" in layers:
            t = time.perf_counter()
            spans = self.presidio.detect(text, language)
            out["patterns"] = [x for x in spans if "spacy" not in x.sources]
            if "spacy" in layers:
                out["spacy"] = [x for x in spans if "spacy" in x.sources]
            timings["patterns+spacy"] = time.perf_counter() - t
        for name in layers:
            if name in ("patterns", "spacy", "llm"):
                continue
            t = time.perf_counter()
            out[name] = self.loaded[name].detect(text)
            timings[name] = time.perf_counter() - t
        return out

    def finalize(self, text: str, raw_by_layer: dict[str, list[Span]], opts: DetectionOptions,
                 layers: list[str] | None = None, timings: dict[str, float] | None = None,
                 use_llm: bool = False, llm=None) -> DetectionResult:
        """Normalise, apply lists/presets/table rules, optionally run the LLM pass,
        merge. Separate from collect() so cached layer outputs can be re-merged
        (benchmarks evaluate many layer combinations without re-running models)."""
        s = self.settings
        timings = timings if timings is not None else {}
        preset = resolve_preset(opts.preset, s.default_preset, text)
        ts = s.timesheet_for(preset, opts.timesheet)
        languages = [opts.language] if opts.language else sorted({b.language for b in self.presidio.blocks(text)})
        raw = [Span(x.start, x.end, x.entity, x.score, set(x.sources), x.text)
               for spans in raw_by_layer.values() for x in spans]

        allow = self.allow | {a.lower() for a in opts.allow_list}
        ignored = self._ignored(opts, ts)
        raw = normalise(raw, s, text, allow, ignored, self.species)
        raw.extend(self._deny_spans(text, opts.deny_list))  # explicit terms bypass all filters
        if preset == "timesheet":
            table_spans = propagate_columns(text, self.merger.merge(list(raw), text), ignored)
            raw.extend(normalise(table_spans, s, text, allow, ignored, self.species))

        verdict_count = 0
        warnings: list[str] = []
        if use_llm:
            t = time.perf_counter()
            raw, verdict_count = self._llm_pass(text, raw, allow, ignored, warnings, llm or self.loaded["llm"])
            timings["llm"] = time.perf_counter() - t

        spans = self.merger.merge(raw, text)
        # merge remainders are new fragments: filter them like everything else
        deny = {x.lower() for x in opts.deny_list}
        spans = [x for x in spans if x.entity == "CUSTOM" or x.text.lower() in deny
                 or normalise([x], s, text, allow, ignored, self.species)]
        if opts.entities:
            spans = [x for x in spans if x.entity in set(opts.entities)]
        return DetectionResult(spans, languages, layers or list(raw_by_layer),
                               {k: round(v, 3) for k, v in timings.items()}, verdict_count, warnings, preset, ts)

    def _ignored(self, opts: DetectionOptions, ts: Timesheet) -> set[str]:
        ignored = set(self.settings.ignored_entities) | set(opts.ignore_entities or [])
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
                  warnings: list[str], llm) -> tuple[list[Span], int]:
        lo, hi = self.settings.layers.llm.adjudicate_range
        prelim = self.merger.merge(raw, text, apply_threshold=False)
        # Pattern/dictionary hits are never put to a vote: the LLM judges "personal
        # data" narrowly (e.g. it calls project codes harmless) and must not veto them.
        uncertain = [x for x in prelim if lo <= x.score < hi and "patterns" not in x.sources]
        uncertain = sorted(uncertain, key=lambda x: -x.score)[: self.settings.layers.llm.max_candidates]
        uncertain.sort(key=lambda x: x.start)
        known = [x for x in prelim if x.score >= hi]
        found, verdicts = llm.detect(text, uncertain, known, warnings)
        found = normalise(found, self.settings, text, allow, ignored, self.species)
        for v in verdicts:
            c = v.span
            if v.is_pii:
                found.append(Span(c.start, c.end, v.entity if v.entity and v.entity != "OTHER_PII" else c.entity,
                                  0.85, {"llm"}, c.text))
            elif getattr(llm.cfg, "allow_veto", True):
                # LLM says "not PII": damp every raw span behind this candidate
                for r in raw:
                    if r.entity == c.entity and r.overlaps(c):
                        r.score *= 0.5
        return raw + found, len(verdicts)
