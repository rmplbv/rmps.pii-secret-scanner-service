"""HTML support: analyse only visible text (and selected attribute values),
write replacements back into the original markup so structure stays intact.

The extracted "virtual text" has a character-exact mapping to the raw HTML:
every virtual character points to the raw slice it came from (an entity such
as "&amp;" is one virtual character backed by five raw characters).
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field

BLOCK = {
    "address", "article", "aside", "blockquote", "br", "dd", "div", "dl", "dt", "fieldset", "figcaption",
    "figure", "footer", "form", "h1", "h2", "h3", "h4", "h5", "h6", "header", "hr", "li", "main", "nav",
    "ol", "p", "pre", "section", "table", "tbody", "td", "tfoot", "th", "thead", "title", "tr", "ul",
    "body", "html", "head", "option", "select", "textarea", "label", "button",
}
_TOKEN = re.compile(r"<!--.*?-->|<!\[CDATA\[.*?\]\]>|<![^>]*>|<\?[^>]*\?>|</?[A-Za-z][^>]*>", re.DOTALL)
_TAGNAME = re.compile(r"</?\s*([A-Za-z][\w:-]*)")
_ATTR = re.compile(r"""([^\s"'<>/=]+)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'>]+))""")
_ENTITY = re.compile(r"&(?:#\d+|#[xX][0-9a-fA-F]+|[A-Za-z][A-Za-z0-9]*);?")


@dataclass
class VirtualText:
    text: str
    # for virtual char i: raw [start, end)
    raw_start: list[int] = field(default_factory=list)
    raw_end: list[int] = field(default_factory=list)


def looks_like_html(text: str) -> bool:
    head = text[:2000].lower()
    return bool(re.search(r"<(?:!doctype|html|body|div|p|span|table|br|a\s|h[1-6]|ul|li|td)\b", head))


def extract(raw: str, attributes: list[str], skip_elements: list[str]) -> VirtualText:
    vt = VirtualText("")
    chars: list[str] = []
    attrs = {a.lower() for a in attributes}
    skip = {e.lower() for e in skip_elements}

    def sep(s: str = "\n") -> None:
        if chars and chars[-1] != "\n":
            for ch in s:
                chars.append(ch)
                vt.raw_start.append(-1)
                vt.raw_end.append(-1)

    def add_text(start: int, end: int) -> None:
        pos = start
        for m in _ENTITY.finditer(raw, start, end):
            _add_plain(pos, m.start())
            decoded = html.unescape(m.group())
            for ch in decoded:
                chars.append(ch)
                vt.raw_start.append(m.start())
                vt.raw_end.append(m.end())
            pos = m.end()
        _add_plain(pos, end)

    def _add_plain(a: int, b: int) -> None:
        for i in range(a, b):
            chars.append(raw[i])
            vt.raw_start.append(i)
            vt.raw_end.append(i + 1)

    pos = 0
    skipping: str | None = None
    attr_segments: list[tuple[int, int]] = []
    for m in _TOKEN.finditer(raw):
        if skipping is None and m.start() > pos:
            add_text(pos, m.start())
        tok = m.group()
        pos = m.end()
        if tok.startswith("<!--"):
            if skipping is None:  # invisible, but shipped with the page: may hold PII
                sep("\n\n")
                add_text(m.start() + 4, m.end() - 3)
                sep("\n\n")
            continue
        if tok.startswith("<!") or tok.startswith("<?"):
            continue
        name_m = _TAGNAME.match(tok)
        name = name_m.group(1).lower() if name_m else ""
        closing = tok.startswith("</")
        if skipping:
            if closing and name == skipping:
                skipping = None
            continue
        if not closing and name in skip and not tok.rstrip().endswith("/>"):
            skipping = name
            continue
        if name in BLOCK:
            sep()
        if not closing:
            for a in _ATTR.finditer(tok):
                if a.group(1).lower() in attrs:
                    g = next(i for i in (2, 3, 4) if a.group(i) is not None)
                    # appended after the visible text, so sentences are not torn apart
                    attr_segments.append((m.start() + a.start(g), m.start() + a.end(g)))
    if skipping is None and pos < len(raw):
        add_text(pos, len(raw))
    for a, b in attr_segments:
        sep("\n\n")
        add_text(a, b)
    vt.text = "".join(chars)
    return vt


def raw_pieces(vt: VirtualText, start: int, end: int) -> list[tuple[int, int]]:
    """Raw intervals covering virtual [start, end), merged where contiguous.
    A span crossing a tag (e.g. "Max <b>Müller</b>") yields several pieces."""
    pieces: list[list[int]] = []
    for i in range(start, end):
        a, b = vt.raw_start[i], vt.raw_end[i]
        if a < 0:
            continue
        if pieces and a <= pieces[-1][1]:
            pieces[-1][1] = max(pieces[-1][1], b)
        else:
            pieces.append([a, b])
    return [(a, b) for a, b in pieces]


def to_raw_span(vt: VirtualText, start: int, end: int) -> tuple[int, int] | None:
    pieces = raw_pieces(vt, start, end)
    return (pieces[0][0], pieces[-1][1]) if pieces else None


def apply(raw: str, vt: VirtualText, replacements: list[tuple[int, int, str]]) -> str:
    """replacements: (virtual_start, virtual_end, new_text). The new text goes
    into the first raw piece, other pieces are emptied; markup between them stays."""
    edits: list[tuple[int, int, str]] = []
    for start, end, new in replacements:
        pieces = raw_pieces(vt, start, end)
        for k, (a, b) in enumerate(pieces):
            edits.append((a, b, html.escape(new, quote=True) if k == 0 else ""))
    out, pos = [], 0
    for a, b, new in sorted(edits):
        if a < pos:
            continue
        out.append(raw[pos:a])
        out.append(new)
        pos = b
    out.append(raw[pos:])
    return "".join(out)
