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
    name = "transformer"

    def __init__(self, settings: Settings):
        from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

        cfg = settings.layers.transformer
        self.cfg = cfg
        tok = AutoTokenizer.from_pretrained(cfg.model)
        model = AutoModelForTokenClassification.from_pretrained(cfg.model)
        self.pipe = pipeline("token-classification", model=model, tokenizer=tok,
                             aggregation_strategy="simple", device=-1)

    def detect(self, text: str, language: str | None = None) -> list[Span]:
        chunks = [c for c in chunk_text(text, self.cfg.chunk_chars, self.cfg.overlap_chars) if c.text.strip()]
        if not chunks:
            return []
        results = self.pipe([c.text for c in chunks], batch_size=self.cfg.batch_size)
        spans: list[Span] = []
        for chunk, ents in zip(chunks, results, strict=True):
            for e in ents:
                label = e.get("entity_group", "")
                entity = self.cfg.label_map.get(label)
                if not entity or e["score"] < self.cfg.min_score:
                    continue
                s, t = chunk.start + int(e["start"]), chunk.start + int(e["end"])
                spans.append(Span(s, t, entity, float(e["score"]), {self.name}, text[s:t]))
        return _merge_adjacent(_dedupe(spans), text)


class GlinerLayer:
    name = "gliner"

    def __init__(self, settings: Settings):
        from gliner import GLiNER

        self.cfg = settings.layers.gliner
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
