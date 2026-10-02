"""Benchmark inputs.

Three kinds of input, mixable:

* ``*.jsonl``: one document per line, like eval/generate.py writes:
  {"id", "text", "spans": [{"start","end","entity"}], optional "format", "preset", "kind", "language"}
* annotated files (``*.txt``, ``*.md``, ``*.html``, ``*.htm``) with inline gold labels::

      Hallo [[PERSON|Anna Huber]], bitte an [[EMAIL_ADDRESS|anna@example.org]] ...

  The markup is removed; the labels become gold spans. A file is "annotated" when it
  contains at least one ``[[ENTITY|...]]``. Put ``<!-- preset: timesheet -->`` or a first
  line ``#preset: timesheet`` in a file to state the expected preset (optional).
* plain files without markup: unlabeled. They are still run through every candidate so
  outputs can be compared side by side (report.html); no recall/precision for them.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

ANNOT = re.compile(r"\[\[([A-Z][A-Z0-9_]*)\|(.*?)\]\]", re.DOTALL)
TEXT_EXT = {".txt", ".md", ".html", ".htm", ".eml", ".csv", ".tsv"}


@dataclass
class Doc:
    id: str
    text: str
    gold: list[dict] | None = None          # None = unlabeled
    format: str = "auto"
    preset: str | None = None               # expected preset (only reported)
    kind: str = "file"
    language: str | None = None
    meta: dict = field(default_factory=dict)

    @property
    def labeled(self) -> bool:
        return self.gold is not None


def strip_annotations(raw: str) -> tuple[str, list[dict] | None]:
    if not ANNOT.search(raw):
        return raw, None
    out, spans, pos, n = [], [], 0, 0
    for m in ANNOT.finditer(raw):
        out.append(raw[pos:m.start()])
        n += m.start() - pos
        value = m.group(2)
        spans.append({"start": n, "end": n + len(value), "entity": m.group(1)})
        out.append(value)
        n += len(value)
        pos = m.end()
    out.append(raw[pos:])
    return "".join(out), spans


def _file_preset(text: str) -> str | None:
    m = re.search(r"(?:<!--\s*preset:\s*(\w+)\s*-->|^#preset:\s*(\w+)\s*$)", text[:500], re.M)
    return (m.group(1) or m.group(2)) if m else None


def load(paths: list[str], limit: int = 0) -> list[Doc]:
    docs: list[Doc] = []
    for p in paths:
        path = Path(p)
        files = sorted(x for x in path.rglob("*") if x.is_file()) if path.is_dir() else [path]
        for f in files:
            if f.suffix == ".jsonl":
                for line in f.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    docs.append(Doc(id=f"{f.stem}:{r.get('id', len(docs))}", text=r["text"], gold=r.get("spans"),
                                    format=r.get("format", "auto"), preset=r.get("preset"),
                                    kind=r.get("kind", "jsonl"), language=r.get("language")))
            elif f.suffix.lower() in TEXT_EXT:
                raw = f.read_text(encoding="utf-8", errors="replace")
                preset = _file_preset(raw)
                if preset and raw.startswith("#preset:"):
                    raw = raw.split("\n", 1)[1] if "\n" in raw else ""
                text, gold = strip_annotations(raw)
                fmt = "html" if f.suffix.lower() in (".html", ".htm") else "auto"
                docs.append(Doc(id=str(f), text=text, gold=gold, format=fmt, preset=preset,
                                kind=f.suffix.lower().lstrip(".")))
    return docs[:limit] if limit else docs
