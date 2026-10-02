"""Apply anonymization operators to detected spans, and reverse them."""

from __future__ import annotations

import hashlib
import hmac
import html
import re
from bisect import bisect_right
from dataclasses import dataclass

from ..config import Settings, Timesheet
from ..models import FAKE_FEATURE, Span
from .fakers import FakeFactory
from .session import SessionState

MODES = ("redact", "placeholder", "mask", "fake", "hash", "keep")
REVERSIBLE_OPS = {"placeholder", "fake", "hash"}


@dataclass
class Replacement:
    span: Span
    operator: str
    replacement: str
    new_start: int
    new_end: int
    reversible: bool

    def to_dict(self, include_original: bool = False) -> dict:
        d = self.span.to_dict(include_text=include_original)
        d.update(operator=self.operator, replacement=self.replacement,
                 new_start=self.new_start, new_end=self.new_end, reversible=self.reversible)
        return d


class Anonymizer:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.cfg = settings.anonymize

    def anonymize(self, text: str, spans: list[Span], state: SessionState, mode: str | None = None,
                  operators: dict[str, str] | None = None, timesheet: Timesheet | None = None,
                  blocks: list[tuple[int, int, str]] | None = None) -> tuple[str, list[Replacement]]:
        mode = mode or self.cfg.default_mode
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
        operators = operators or {}
        ts = timesheet or self.settings.timesheet
        fakes = FakeFactory(self.settings, state, ts, text)
        starts = [b[0] for b in blocks or []]

        def lang_at(pos: int) -> str:
            if not blocks:
                return self.settings.default_language
            return blocks[max(0, bisect_right(starts, pos) - 1)][2]

        out: list[str] = []
        reps: list[Replacement] = []
        cursor = 0
        new_len = 0
        for span in sorted(spans, key=lambda s: s.start):
            if span.start < cursor:  # defensive: overlapping input
                continue
            op = self._operator(span.entity, mode, operators, ts)
            original = text[span.start:span.end]
            replacement, op = self._apply(op, span.entity, original, state, fakes, lang_at(span.start))
            out.append(text[cursor:span.start])
            new_len += span.start - cursor
            out.append(replacement)
            reps.append(Replacement(span, op, replacement, new_len, new_len + len(replacement),
                                    op in REVERSIBLE_OPS))
            new_len += len(replacement)
            cursor = span.end
        out.append(text[cursor:])
        return "".join(out), reps

    def _operator(self, entity: str, mode: str, operators: dict[str, str], ts: Timesheet) -> str:
        op = operators.get(entity) or operators.get("*") or mode
        if op == "fake":
            if entity in self.cfg.no_fake_entities:
                return "placeholder"
            feature = FAKE_FEATURE.get(entity)
            if feature and not getattr(ts, feature).fake:
                return "placeholder"
        return op

    def _apply(self, op: str, entity: str, original: str, state: SessionState,
               fakes: FakeFactory, language: str) -> tuple[str, str]:
        if op == "keep":
            return original, op
        if op == "redact":
            return self.cfg.redact_text, op
        if op == "mask":
            keep = self.cfg.mask_keep_last if entity in self.cfg.mask_keep_last_entities else 0
            return mask(original, self.cfg.mask_char, keep), op
        known = state.lookup(f"{op}:{entity}", original)
        if known is not None:
            return known, op
        if op == "fake":
            value = fakes.fake(entity, original, language)
            if value is None:  # no faker for this entity / unparseable value
                return self._apply("placeholder", entity, original, state, fakes, language)
        elif op == "hash":
            digest = hmac.new(state.seed.encode(), f"{entity}\x00{original.casefold()}".encode(),
                              hashlib.sha256).hexdigest()[:10]
            value = self.cfg.hash_format.format(entity=entity, digest=digest)
        elif op == "placeholder":
            value = self.cfg.placeholder_format.format(entity=entity, n=state.next_number(entity))
        else:
            raise ValueError(f"unknown operator {op!r}")
        state.remember(f"{op}:{entity}", original, value)
        return value, op


def mask(value: str, char: str = "*", keep_last: int = 4) -> str:
    alnum = [i for i, c in enumerate(value) if c.isalnum()]
    keep = set(alnum[-keep_last:]) if keep_last and len(alnum) > 2 * keep_last else set()
    return "".join(c if (not c.isalnum() or i in keep) else char for i, c in enumerate(value))


def deanonymize(text: str, state: SessionState, person_tokens: bool = True) -> tuple[str, int]:
    """Replace every known replacement in `text` by its original. Works on text
    produced elsewhere (e.g. an LLM answer about the pseudonymized text).

    Full replacements match case-insensitively; single fake name tokens (for
    surname-only mentions) match case-sensitively (as written or upper case),
    so ordinary lower-case words are never touched."""
    variants: dict[str, str] = {}
    for fake, orig in state.reverse.items():
        for v in (fake, fake.lower(), fake.upper(), fake.title()):
            variants.setdefault(v, orig)
        escaped = html.escape(fake, quote=False)
        if escaped != fake:  # placeholder inside HTML: "&lt;PERSON_1&gt;"
            variants.setdefault(escaped, html.escape(orig, quote=False))
    if person_tokens:
        for fake_tok, orig_tok in state.token_reverse.items():
            if len(fake_tok) >= 3:
                variants.setdefault(fake_tok, orig_tok)
                variants.setdefault(fake_tok.upper(), orig_tok.upper())
    if not variants:
        return text, 0
    alternatives = []
    for k in sorted(variants, key=len, reverse=True):
        pat = re.escape(k)
        if k[:1].isalnum():
            pat = r"(?<!\w)" + pat
        if k[-1:].isalnum():
            pat = pat + r"(?!\w)"
        alternatives.append(pat)
    rx = re.compile("|".join(alternatives))
    count = 0

    def sub(m: re.Match) -> str:
        nonlocal count
        count += 1
        return variants[m.group()]

    return rx.sub(sub, text), count
