"""Text chunking and language detection."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache


@dataclass
class Chunk:
    start: int
    text: str
    language: str | None = None

    @property
    def end(self) -> int:
        return self.start + len(self.text)


def chunk_text(text: str, size: int, overlap: int) -> list[Chunk]:
    """Split text into overlapping chunks of roughly `size` chars, cutting at
    paragraph, line, sentence or word boundaries where possible."""
    if len(text) <= size:
        return [Chunk(0, text)]
    chunks: list[Chunk] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            end = _best_cut(text, start + size // 2, end)
        chunks.append(Chunk(start, text[start:end]))
        if end >= len(text):
            break
        nxt = max(end - overlap, start + 1)
        # begin the next chunk at a word boundary
        m = re.search(r"\s", text[nxt:end])
        start = nxt + m.end() if m else nxt
    return chunks


def _best_cut(text: str, lo: int, hi: int) -> int:
    window = text[lo:hi]
    for pattern in (r"\n\s*\n", r"\n", r"[.!?;]\s", r"\s"):
        matches = list(re.finditer(pattern, window))
        if matches:
            return lo + matches[-1].end()
    return hi


def split_blocks(text: str, max_chars: int = 5000) -> list[Chunk]:
    """Non-overlapping blocks of whole paragraphs (for per-block language
    detection and spaCy). max_chars=0: one block per paragraph; paragraphs
    longer than 5000 chars are always split."""
    paras: list[tuple[int, int]] = []
    pos = 0
    for m in re.finditer(r"\n\s*\n", text):
        paras.append((pos, m.end()))
        pos = m.end()
    if pos < len(text):
        paras.append((pos, len(text)))
    blocks: list[tuple[int, int]] = []
    for s, e in paras:
        if max_chars and blocks and e - blocks[-1][0] <= max_chars:
            blocks[-1] = (blocks[-1][0], e)
        else:
            blocks.append((s, e))
    out: list[Chunk] = []
    limit = max_chars or 5000
    for s, e in blocks:
        if e - s > limit:
            out.extend(Chunk(s + c.start, c.text) for c in chunk_text(text[s:e], limit, 0))
        else:
            out.append(Chunk(s, text[s:e]))
    return out or [Chunk(0, text)]


@lru_cache(maxsize=1)
def _detector(languages: tuple[str, ...]):
    from lingua import IsoCode639_1, LanguageDetectorBuilder

    codes = [getattr(IsoCode639_1, lang.upper()) for lang in languages]
    return LanguageDetectorBuilder.from_iso_codes_639_1(*codes).build()


def detect_language(text: str, languages: list[str], default: str) -> str:
    if len(languages) == 1:
        return languages[0]
    sample = text[:3000]
    if len(re.findall(r"[^\W\d_]{2,}", sample)) < 3:
        return default
    values = _detector(tuple(languages)).compute_language_confidence_values(sample)
    if not values or values[0].value < 0.6:
        return default
    return values[0].language.iso_code_639_1.name.lower()
