"""Layers L2 (transformer token classification) and L3 (GLiNER zero-shot).
Both are multilingual, so they run on the whole text without language routing."""

from __future__ import annotations

import logging
import re

from ..config import Settings
from ..models import Span
from ..text_utils import chunk_text

log = logging.getLogger(__name__)


class TransformerLayer:

    def __init__(self, settings: Settings, name: str = "transformer", **overrides):
        """`overrides` replace fields of layers.transformer (model, label_map, ...),
        so several token-classification models can run side by side."""
        from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

        cfg = settings.layers.transformer.model_copy(update=overrides)
        self.cfg = cfg
        self.name = name
        tok = AutoTokenizer.from_pretrained(cfg.model, trust_remote_code=getattr(cfg, "trust_remote_code", False))
        model = AutoModelForTokenClassification.from_pretrained(
            cfg.model, trust_remote_code=getattr(cfg, "trust_remote_code", False))
        agg = getattr(cfg, "aggregation", "simple")
        self.bioes = agg == "bioes"  # B/I/E/S tags: the HF aggregation only understands B/I
        self.pipe = pipeline("token-classification", model=model, tokenizer=tok,
                             aggregation_strategy="none" if self.bioes else agg, device=-1)

    def detect(self, text: str, language: str | None = None) -> list[Span]:
        chunks = [c for c in chunk_text(text, self.cfg.chunk_chars, self.cfg.overlap_chars) if c.text.strip()]
        if not chunks:
            return []
        results = self.pipe([c.text for c in chunks], batch_size=self.cfg.batch_size)
        if self.bioes:
            results = [_group_bioes(r, c.text) for r, c in zip(results, chunks, strict=True)]
        spans: list[Span] = []
        for chunk, ents in zip(chunks, results, strict=True):
            for e in ents:
                label = e.get("entity_group") or e.get("entity", "")
                label = re.sub(r"^[BIESL]-", "", label)
                entity = self.cfg.label_map.get(label) or self.cfg.label_map.get(label.lower())
                if not entity or e["score"] < self.cfg.min_score:
                    continue
                s, t = chunk.start + int(e["start"]), chunk.start + int(e["end"])
                spans.append(Span(s, t, entity, float(e["score"]), {self.name}, text[s:t]))
        return _merge_adjacent(_dedupe(spans), text)


class GlinerLayer:

    def __init__(self, settings: Settings, name: str = "gliner", **overrides):
        from gliner import GLiNER

        self.cfg = settings.layers.gliner.model_copy(update=overrides)
        self.name = name
        self.model = GLiNER.from_pretrained(self.cfg.model)
        self.model.eval()

    def detect(self, text: str, language: str | None = None, labels: dict[str, str] | None = None) -> list[Span]:
        labels = labels or self.cfg.labels
        spans: list[Span] = []
        for chunk in _word_chunks(text, self.cfg.chunk_words, self.cfg.overlap_words):
            ents = self.model.predict_entities(chunk[1], list(labels), threshold=self.cfg.threshold)
            for e in ents:
                entity = labels.get(e["label"])
                if not entity:
                    continue
                s, t = chunk[0] + e["start"], chunk[0] + e["end"]
                spans.append(Span(s, t, entity, float(e["score"]), {self.name}, text[s:t]))
        return _dedupe(spans)


def _group_bioes(tokens: list[dict], text: str) -> list[dict]:
    """Group per-token predictions (B-/I-/E-/S-/O) into entity spans."""
    out: list[dict] = []
    for t in tokens:
        tag = t.get("entity", "O")
        if tag == "O" or t.get("start") is None:
            continue
        prefix, _, label = tag.partition("-") if tag[:2] in ("B-", "I-", "E-", "S-") else ("I", "", tag)
        cur = out[-1] if out else None
        joinable = cur is not None and cur["entity_group"] == label and prefix in ("I", "E") \
            and not cur.get("closed") and t["start"] - cur["end"] <= 1
        if joinable:
            cur["end"] = t["end"]
            cur["_scores"].append(t["score"])
        else:
            out.append({"entity_group": label, "start": t["start"], "end": t["end"], "_scores": [t["score"]]})
            cur = out[-1]
        if prefix in ("E", "S"):
            cur["closed"] = True
    for e in out:
        e["score"] = sum(e["_scores"]) / len(e["_scores"])
    return out


def _word_chunks(text: str, size: int, overlap: int) -> list[tuple[int, str]]:
    """GLiNER has a word-based context limit (~384 tokens)."""
    words = list(re.finditer(r"\S+", text))
    if not words:
        return []
    out = []
    i = 0
    step = max(1, size - overlap)
    while i < len(words):
        j = min(i + size, len(words))
        s, e = words[i].start(), words[j - 1].end()
        out.append((s, text[s:e]))
        if j == len(words):
            break
        i += step
    return out


def _dedupe(spans: list[Span]) -> list[Span]:
    """Chunks overlap: collapse identical spans, keeping the best score."""
    best: dict[tuple, Span] = {}
    for s in spans:
        k = (s.start, s.end, s.entity)
        if k not in best or s.score > best[k].score:
            best[k] = s
    return sorted(best.values(), key=lambda s: s.start)


def _merge_adjacent(spans: list[Span], text: str) -> list[Span]:
    """Join same-entity fragments separated only by a space or hyphen
    (subword aggregation sometimes splits names like "Müller-Lüdenscheidt")."""
    out: list[Span] = []
    for s in spans:
        if out and out[-1].entity == s.entity and 0 <= s.start - out[-1].end <= 1 \
                and text[out[-1].end:s.start] in ("", " ", "-"):
            prev = out[-1]
            out[-1] = Span(prev.start, s.end, s.entity, min(prev.score, s.score), prev.sources, text[prev.start:s.end])
        else:
            out.append(s)
    return out


class Gliner2Layer:
    """GLiNER2 / GLiNER2.5 models (package `gliner2`)."""

    def __init__(self, settings: Settings, name: str = "gliner2", **overrides):
        self.cfg = settings.layers.gliner.model_copy(update=overrides)
        self.name = name
        try:  # GLiNER2.5 "boundary" checkpoints need AutoExtractor; it also loads 2.0 span models
            from gliner2 import AutoExtractor
            self.model = AutoExtractor.from_pretrained(self.cfg.model)
        except ImportError:
            from gliner2 import GLiNER2
            self.model = GLiNER2.from_pretrained(self.cfg.model)

    def detect(self, text: str, language: str | None = None) -> list[Span]:
        labels = self.cfg.labels
        spans: list[Span] = []
        for start, chunk in _word_chunks(text, self.cfg.chunk_words, self.cfg.overlap_words):
            res = self.model.extract_entities(chunk, list(labels), include_confidence=True, include_spans=True,
                                              threshold=self.cfg.threshold)
            for label, items in (res.get("entities") or {}).items():
                entity = labels.get(label)
                for it in items or []:
                    if not entity or not isinstance(it, dict):
                        continue
                    if "start" in it and "end" in it:
                        hits = [(it["start"], it["end"])]
                    else:  # text only: locate it
                        hits = [m.span() for m in re.finditer(re.escape(it.get("text", "")), chunk)] if it.get("text") else []
                    for a, b in hits:
                        spans.append(Span(start + a, start + b, entity, float(it.get("confidence", 0.8)),
                                          {self.name}, text[start + a:start + b]))
        return _dedupe(spans)
