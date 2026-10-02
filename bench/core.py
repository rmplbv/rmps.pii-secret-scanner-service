"""Shared benchmark machinery: prepared documents, span cache, layer adapters,
combination evaluation and metrics."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from app.config import Settings
from app.detection.merger import family
from app.formats import html as html_fmt
from app.models import Span

from .data import Doc

BASE_LAYERS = ["patterns", "spacy"]


# -- candidates ------------------------------------------------------------------------
def load_candidates(path: str) -> tuple[dict, list[dict]]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cands = data.get("candidates", [])
    ids = [c["id"] for c in cands]
    dup = {i for i in ids if ids.count(i) > 1}
    if dup:
        raise ValueError(f"duplicate candidate ids: {dup}")
    return data.get("defaults", {}), cands


def build_layer(settings: Settings, cand: dict):
    """Instantiate a detection layer for an NER candidate; its spans carry cand['id'] as source."""
    from app.detection import ml_layers

    kind = cand["kind"]
    common = {k: v for k, v in cand.items() if k in (
        "model", "label_map", "chunk_chars", "overlap_chars", "batch_size", "min_score", "threshold",
        "chunk_words", "overlap_words", "aggregation", "trust_remote_code")}
    if cand.get("labels") not in (None, "default"):
        common["labels"] = cand["labels"]
    if kind == "hf_token":
        return ml_layers.TransformerLayer(settings, name=cand["id"], **common)
    if kind == "gliner":
        return ml_layers.GlinerLayer(settings, name=cand["id"], **common)
    if kind == "gliner2":
        return ml_layers.Gliner2Layer(settings, name=cand["id"], **common)
    raise ValueError(f"unknown NER kind {kind!r}")


# -- prepared documents ------------------------------------------------------------------
@dataclass
class Prepared:
    doc: Doc
    text: str                       # analysed text (virtual text for HTML)
    vt: object | None               # html VirtualText

    def to_raw(self, start: int, end: int) -> tuple[int, int]:
        if self.vt is None:
            return start, end
        r = html_fmt.to_raw_span(self.vt, start, end)
        return r if r else (start, end)


def prepare(docs: list[Doc], settings: Settings) -> list[Prepared]:
    out = []
    for d in docs:
        fmt = d.format
        if fmt == "auto":
            fmt = "html" if html_fmt.looks_like_html(d.text) else "text"
        if fmt == "html":
            vt = html_fmt.extract(d.text, settings.html.attributes, settings.html.skip_elements)
            out.append(Prepared(d, vt.text, vt))
        else:
            out.append(Prepared(d, d.text, None))
    return out


# -- span cache ----------------------------------------------------------------------------
def span_to_json(s: Span) -> dict:
    return {"s": s.start, "e": s.end, "t": s.entity, "p": round(s.score, 4), "src": sorted(s.sources)}


def span_from_json(d: dict, text: str) -> Span:
    return Span(d["s"], d["e"], d["t"], d["p"], set(d["src"]), text[d["s"]:d["e"]])


def write_cache(path: Path, per_doc: dict[str, list[Span]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for doc_id, spans in per_doc.items():
            fh.write(json.dumps({"id": doc_id, "spans": [span_to_json(s) for s in spans]}, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def read_cache(path: Path, prepared: list[Prepared]) -> dict[str, list[Span]]:
    texts = {p.doc.id: p.text for p in prepared}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["id"] in texts:
            out[r["id"]] = [span_from_json(x, texts[r["id"]]) for x in r["spans"]]
    return out


# -- metrics -------------------------------------------------------------------------------
def evaluate(prepared: list[Prepared], preds: dict[str, list[Span]], presets: dict[str, str] | None = None) -> dict:
    """Recall (family-aware), precision (any type), leak rate (uncovered gold chars) —
    overall, per entity and per document kind. Unlabeled docs are skipped."""
    per_ent: dict[str, list[int]] = {}
    per_kind: dict[str, list[int]] = {}
    tp_pred = n_pred = gold_chars = leaked = 0
    fp_examples: dict[str, list[str]] = {}
    miss_examples: dict[str, list[str]] = {}
    preset_ok = preset_n = 0
    for p in prepared:
        d = p.doc
        if not d.labeled:
            continue
        raw_preds = []
        for s in preds.get(d.id, []):
            a, b = p.to_raw(s.start, s.end)
            raw_preds.append((a, b, s.entity))
        covered = bytearray(len(d.text))
        for a, b, _ in raw_preds:
            covered[a:b] = b"\x01" * (b - a)
        k = per_kind.setdefault(d.kind, [0, 0, 0, 0])  # gold chars, leaked chars, gold, hit
        for g in d.gold:
            e = per_ent.setdefault(g["entity"], [0, 0])
            e[0] += 1
            k[2] += 1
            hit = any(a < g["end"] and g["start"] < b and family(t) == family(g["entity"]) for a, b, t in raw_preds)
            if hit:
                e[1] += 1
                k[3] += 1
            else:
                miss_examples.setdefault(g["entity"], []).append(d.text[g["start"]:g["end"]])
            chars = [i for i in range(g["start"], g["end"]) if not d.text[i].isspace()]
            gold_chars += len(chars)
            lk = sum(1 for i in chars if not covered[i])
            leaked += lk
            k[0] += len(chars)
            k[1] += lk
        for a, b, t in raw_preds:
            n_pred += 1
            if any(a < g["end"] and g["start"] < b for g in d.gold):
                tp_pred += 1
            else:
                fp_examples.setdefault(t, []).append(d.text[a:b])
        if presets and d.preset and d.id in presets:
            preset_n += 1
            preset_ok += presets[d.id] == d.preset
    gold_total = sum(v[0] for v in per_ent.values())
    return {
        "recall": round(sum(v[1] for v in per_ent.values()) / max(1, gold_total), 4),
        "precision": round(tp_pred / max(1, n_pred), 4),
        "leak_rate": round(leaked / max(1, gold_chars), 4),
        "gold": gold_total, "predicted": n_pred,
        "per_entity": {k: {"recall": round(v[1] / v[0], 3), "n": v[0]} for k, v in sorted(per_ent.items())},
        "per_kind": {k: {"leak_rate": round(v[1] / max(1, v[0]), 4), "recall": round(v[3] / max(1, v[2]), 3)}
                     for k, v in sorted(per_kind.items())},
        "false_positives": {k: _top(v) for k, v in sorted(fp_examples.items(), key=lambda kv: -len(kv[1]))},
        "missed": {k: _top(v) for k, v in sorted(miss_examples.items(), key=lambda kv: -len(kv[1]))},
        "preset_accuracy": round(preset_ok / preset_n, 3) if preset_n else None,
    }


def _top(values: list[str], n: int = 8) -> dict:
    counts: dict[str, int] = {}
    for v in values:
        v = re.sub(r"\s+", " ", v)[:60]
        counts[v] = counts.get(v, 0) + 1
    return {"count": len(values), "examples": dict(sorted(counts.items(), key=lambda kv: -kv[1])[:n])}


def peak_rss_gb(pid: int | str = "self") -> float | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return round(int(line.split()[1]) / 1024 / 1024, 2)
    except OSError:
        return None
    return None
