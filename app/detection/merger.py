"""Ensemble merging of spans from several detection layers.

1. Normalise entity names, drop ignored entities and allow-listed text.
2. Every span is a candidate. Its score is its own score (weighted by the
   reliability of its source) plus a boost for every *other* layer that found an
   overlapping span of the same entity type.
3. Candidates are selected greedily by (score, priority, length) without overlap.
4. Recall bias: a rejected span whose remainder (the part not covered by the
   selected spans) is still substantial is kept as its own span, so sloppy
   boundaries never leak text that some layer considered PII.
5. Thresholds: candidates found by a single layer need a higher score.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from ..config import Settings
from ..models import Span

FAMILIES = {
    "LOCATION": "PLACE", "ADDRESS": "PLACE", "POSTAL_CODE": "PLACE",
    "ID_NUMBER": "ID", "EMPLOYEE_ID": "ID", "USERNAME": "ID",
    "PROJECT_NAME": "PROJECT", "PROJECT_CODE": "PROJECT",
    "DATE_TIME": "DATE", "DATE_OF_BIRTH": "DATE",
    "RELIGION": "SPECIAL", "POLITICAL_OPINION": "SPECIAL", "TRADE_UNION": "SPECIAL",
    "ETHNIC_ORIGIN": "SPECIAL", "SEXUAL_ORIENTATION": "SPECIAL", "GENETIC_BIOMETRIC": "SPECIAL",
    "HEALTH_DATA": "SPECIAL",
}


def family(entity: str) -> str:
    """Entity types that different layers use for the same thing agree with each other."""
    if entity[:3] in ("DE_", "US_", "UK_"):
        return "ID"
    return FAMILIES.get(entity, entity)


DEFAULT_SOURCE_WEIGHTS = {"patterns": 1.0, "transformer": 0.95, "gliner": 0.9, "llm": 0.9, "spacy": 0.85}


# Words that are never PII on their own (NER models like to tag pronouns).
NON_PII_WORDS = {
    "ich", "du", "er", "sie", "es", "wir", "ihr", "ihm", "ihn", "ihnen", "mich", "mir", "dich", "dir",
    "uns", "euch", "man", "jemand", "i", "me", "you", "he", "him", "she", "her", "it", "we", "us",
    "they", "them", "someone", "herr", "frau", "mr", "mrs", "ms", "dr", "prof", "team", "kunde", "customer",
    # greetings / sign-offs that NER models tag at the start of a line
    "hallo", "hi", "hello", "dear", "liebe", "lieber", "best", "regards", "cheers", "thanks", "danke",
    "grüße", "gruß", "viele grüße", "beste grüße", "mfg", "vg", "lg",
}
HONORIFIC = re.compile(
    r"^(?:(?:Herrn?|Frau|Fr\.|Hr\.|Dr\.|Prof\.|Dipl\.-Ing\.|Mr\.?|Mrs\.?|Ms\.?|Miss|Mx\.?|Sir|Dame)\s+)+")


def normalise(spans: Iterable[Span], settings: Settings, text: str,
              allow: set[str] | None = None, ignored: set[str] | None = None) -> list[Span]:
    aliases = settings.entity_aliases
    ignored = set(settings.ignored_entities) if ignored is None else ignored
    allow = allow or set()
    out = []
    for s in spans:
        s.entity = aliases.get(s.entity, s.entity)
        if s.entity in ignored:
            continue
        # trim whitespace/punctuation at the edges
        while s.start < s.end and text[s.start] in " \t\n\r,;:([\"'":
            s.start += 1
        while s.end > s.start and text[s.end - 1] in " \t\n\r,;:)]\"'":
            s.end -= 1
        if s.end - s.start < 2 and not text[s.start:s.end].isdigit():
            continue
        if s.entity == "PERSON":
            m = HONORIFIC.match(text[s.start:s.end])
            if m and m.end() < s.end - s.start:
                s.start += m.end()
        s.text = text[s.start:s.end]
        low = s.text.lower().rstrip(".")
        if low in allow or low in NON_PII_WORDS:
            continue
        out.append(s)
    return out


class Merger:
    def __init__(self, settings: Settings):
        m = settings.merge
        self.settings = settings
        self.min_score = m.min_score
        self.single_min = m.single_source_min_score
        self.boost = m.agreement_boost
        self.priority = m.entity_priority
        extra = settings.raw.get("merge", {})
        self.weights = {**DEFAULT_SOURCE_WEIGHTS, **extra.get("source_weights", {})}
        self.entity_min = extra.get("entity_min_score", {})
        self.keep_remainders = extra.get("keep_remainders", True)

    def _threshold(self, span: Span) -> float:
        if span.entity in self.entity_min:
            return self.entity_min[span.entity]
        return self.min_score if len(span.sources) > 1 else self.single_min

    def score_candidates(self, spans: list[Span]) -> list[Span]:
        """Return one candidate per input span with agreement-adjusted score and
        the full set of agreeing sources."""
        cands = []
        spans = sorted(spans, key=lambda s: (s.start, s.end))
        for i, s in enumerate(spans):
            agreeing = set(s.sources)
            for j, o in enumerate(spans):
                if o.start >= s.end:
                    break
                if i != j and family(o.entity) == family(s.entity) and o.overlaps(s):
                    agreeing |= o.sources
            own = max(self.weights.get(src, 0.8) for src in s.sources) * s.score
            score = own + self.boost * (len(agreeing) - len(s.sources))
            c = Span(s.start, s.end, s.entity, score, agreeing, s.text)
            c.origin = frozenset(s.sources)  # layers that produced this exact span/type
            cands.append(c)
        return cands

    def merge(self, spans: list[Span], text: str, apply_threshold: bool = True) -> list[Span]:
        cands = self.score_candidates(spans)
        # Identical boundaries: keep one candidate. Prefer the type a pattern or
        # dictionary recognizer assigned (precise, e.g. "ver.di" -> TRADE_UNION
        # rather than ORGANIZATION), else the best score.
        uniq: dict[tuple, Span] = {}
        for c in cands:
            key = (c.start, c.end)
            prev = uniq.get(key)
            if prev is None:
                uniq[key] = c
                continue
            winner, loser = (c, prev) if self._better_type(c, prev) else (prev, c)
            winner.sources |= loser.sources
            winner.score = max(winner.score, loser.score)
            winner.origin = winner.origin | loser.origin
            uniq[key] = winner
        cands = list(uniq.values())
        if apply_threshold:
            cands = [c for c in cands if c.score >= self._threshold(c)]
        # Scores are uncapped while selecting, so that a precise span backed by
        # several layers beats a sloppy long span that merely overlaps them.
        cands.sort(key=lambda c: (-c.score, -self.priority.get(c.entity, 10), -(c.end - c.start)))
        selected: list[Span] = []
        rejected: list[Span] = []
        for c in cands:
            if any(c.overlaps(s) for s in selected):
                rejected.append(c)
            else:
                selected.append(c)
        selected = self._absorb(rejected, selected)
        if self.keep_remainders:
            selected.extend(self._remainders(rejected, selected, text))
        for c in selected:
            c.score = min(1.0, c.score)
        return sorted(selected, key=lambda s: s.start)

    @staticmethod
    def _better_type(a: Span, b: Span) -> bool:
        pa = "patterns" in getattr(a, "origin", a.sources)
        pb = "patterns" in getattr(b, "origin", b.sources)
        if pa != pb:
            return pa
        return a.score > b.score

    def _absorb(self, rejected: list[Span], selected: list[Span]) -> list[Span]:
        """A rejected span of higher priority and the same family that fully
        contains selected spans replaces them (e.g. ADDRESS "Hauptstraße 5,
        80331 München" absorbs LOCATION "Hauptstraße" and "München")."""
        for r in sorted(rejected, key=lambda c: -(c.end - c.start)):
            inner = [s for s in selected if s.overlaps(r)]
            if not inner:
                continue
            fam = family(r.entity)
            prio = self.priority.get(r.entity, 10)
            if all(family(s.entity) == fam and r.start <= s.start and s.end <= r.end
                   and self.priority.get(s.entity, 10) < prio for s in inner):
                for s in inner:
                    r.sources |= s.sources
                    r.score = max(r.score, s.score)
                selected = [s for s in selected if s not in inner] + [r]
        return selected

    def _remainders(self, rejected: list[Span], selected: list[Span], text: str) -> list[Span]:
        out: list[Span] = []
        for r in rejected:
            if any(family(s.entity) == family(r.entity) and s.overlaps(r) for s in selected):
                continue  # same entity already covers it with better boundaries
            covered = sorted((s for s in selected + out if s.overlaps(r)), key=lambda s: s.start)
            pos = r.start
            pieces = []
            for s in covered:
                if s.start > pos:
                    pieces.append((pos, s.start))
                pos = max(pos, s.end)
            if pos < r.end:
                pieces.append((pos, r.end))
            for a, b in pieces:
                frag = text[a:b]
                m = re.search(r"\w[\w\W]*\w|\w", frag)
                if not m or len(re.sub(r"\W", "", m.group())) < 3:
                    continue
                a, b = a + m.start(), a + m.end()
                out.append(Span(a, b, r.entity, r.score, set(r.sources), text[a:b]))
        return out
