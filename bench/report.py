"""Render bench results (scores.json + resources.json) as Markdown and HTML."""

from __future__ import annotations

import html
import json
from pathlib import Path

ENT_ORDER = ["PERSON", "ORGANIZATION", "LOCATION", "ADDRESS", "EMAIL_ADDRESS", "PHONE_NUMBER", "IBAN_CODE",
             "DATE_OF_BIRTH", "EMPLOYEE_ID", "PROJECT_NAME", "PROJECT_CODE", "DATE_TIME", "CLOCK_TIME",
             "DURATION", "HEALTH_DATA", "RELIGION", "TRADE_UNION", "POLITICAL_OPINION"]


def _pct(x) -> str:
    return "—" if x is None else f"{100 * x:.1f}%"


def _f(x, d=3) -> str:
    return "—" if x is None else f"{x:.{d}f}"


def _ram(res: dict, layers: list[str]) -> str:
    vals = [res.get(l if not l.startswith("llm:") else l[4:], {}).get("peak_rss_gb") for l in layers
            if l not in ("patterns", "spacy")]
    base = res.get("patterns+spacy", {}).get("peak_rss_gb")
    vals = [v for v in vals if v]
    if base:
        vals.append(base)
    return f"{sum(vals):.1f}" if vals else "—"


def _time(res: dict, layers: list[str]) -> str:
    t = res.get("patterns+spacy", {}).get("s_per_doc", 0) or 0
    for l in layers:
        if l in ("patterns", "spacy"):
            continue
        t += res.get(l if not l.startswith("llm:") else l[4:], {}).get("s_per_doc", 0) or 0
    return f"{t:.2f}"


def write_report(out: Path) -> None:
    out = Path(out)
    scores = json.loads((out / "scores.json").read_text())
    res = json.loads((out / "resources.json").read_text()) if (out / "resources.json").exists() else {}
    meta = json.loads((out / "meta.json").read_text()) if (out / "meta.json").exists() else {}
    full = {k: v for k, v in scores.items() if not v.get("llm_subset")}
    llm = {k: v for k, v in scores.items() if v.get("llm_subset")}
    ranked = sorted(full.values(), key=lambda v: (v["leak_rate"], -v["precision"], len(v["layers"])))
    combos = [v for v in ranked if not v["solo"]]
    solos = sorted((v for v in ranked if v["solo"]), key=lambda v: v["leak_rate"])

    md = [f"# PII benchmark: {out.name}", "",
          f"Documents: {meta.get('docs', '?')} ({meta.get('labeled', '?')} labeled), data: {meta.get('data')}", "",
          "*Leak rate* = share of gold PII characters not covered by any prediction (lower is better). "
          "*Recall* counts a gold span as found when a prediction of the same entity family overlaps it. "
          "*RAM* = peak resident memory of the processes involved (models loaded separately; in production "
          "shared libraries make the sum somewhat smaller). Time = seconds per document on this machine.", ""]

    md += ["## Ensembles (patterns + spaCy + NER models)", "",
           "| # | layers | leak rate | recall | precision | preset acc. | RAM GB | s/doc |", "|---|---|---|---|---|---|---|---|"]
    for i, v in enumerate(combos[:25], 1):
        tag = " **(current)**" if v.get("baseline") else ""
        md.append(f"| {i} | {' + '.join(v['layers'])}{tag} | {_pct(v['leak_rate'])} | {_f(v['recall'])} | "
                  f"{_f(v['precision'])} | {_pct(v.get('preset_accuracy'))} | {_ram(res, v['layers'])} | {_time(res, v['layers'])} |")

    md += ["", "## NER models", "",
           "| model | kind | alone: leak / recall / precision | with patterns+spaCy: leak / recall / precision | load s | s/doc | peak RAM GB |",
           "|---|---|---|---|---|---|---|"]
    by_layers = {tuple(v["layers"]): v for v in full.values()}
    for v in solos:
        cid = v["layers"][0]
        r = res.get(cid, {})
        ens = by_layers.get(("patterns", "spacy", cid), {})
        md.append(f"| {cid} | {r.get('kind', '')} | {_pct(v['leak_rate'])} / {_f(v['recall'])} / {_f(v['precision'])} | "
                  f"{_pct(ens.get('leak_rate'))} / {_f(ens.get('recall'))} / {_f(ens.get('precision'))} | "
                  f"{r.get('load_s', '—')} | {r.get('s_per_doc', '—')} | {r.get('peak_rss_gb', '—')} |")
    failed = [k for k, r in res.items() if r.get("error")]
    if failed:
        md += ["", "Failed: " + ", ".join(f"{k} ({res[k]['error']})" for k in failed)]

    if llm:
        some = next(iter(llm.values()))
        md += ["", f"## LLM layer (on top of `{' + '.join(some['layers'][:-1])}`)", "",
               "Each LLM is compared with the same ensemble *without* LLM on the same documents "
               "(the first N of the dataset; dense large models get fewer).", "",
               "| LLM | docs | leak rate | Δ leak | recall | Δ recall | precision | Δ precision | GGUF GB | peak RAM GB | prompt tok/s | gen tok/s | s/doc | warnings |",
               "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for v in sorted(llm.values(), key=lambda v: (v["leak_rate"] - v["base"]["leak_rate"], -v["precision"])):
            r = res.get(v["llm"], {})
            b = v["base"]
            md.append(f"| {v['llm']} | {v['docs']} | {_pct(v['leak_rate'])} | {100 * (v['leak_rate'] - b['leak_rate']):+.2f} pp | "
                      f"{_f(v['recall'])} | {v['recall'] - b['recall']:+.3f} | {_f(v['precision'])} | "
                      f"{v['precision'] - b['precision']:+.3f} | {r.get('gguf_gb', '—')} | {r.get('peak_rss_gb', '—')} | "
                      f"{r.get('prompt_tok_s', '—')} | {r.get('gen_tok_s', '—')} | {r.get('s_per_doc', '—')} | {r.get('warnings', '—')} |")
    failed_llm = [k for k, r in res.items() if r.get("kind") == "llm" and r.get("error")]
    if failed_llm:
        md += ["", "Failed LLMs: " + ", ".join(f"{k} ({res[k]['error']})" for k in failed_llm)]

    top = combos[:6]
    if top:
        ents = [e for e in ENT_ORDER if any(e in v["per_entity"] for v in top)]
        ents += sorted({e for v in top for e in v["per_entity"]} - set(ents))
        md += ["", "## Recall per entity (top ensembles)", "",
               "| entity | n | " + " | ".join(f"#{i}" for i in range(1, len(top) + 1)) + " |",
               "|---|---|" + "---|" * len(top)]
        for e in ents:
            n = top[0]["per_entity"].get(e, {}).get("n", "")
            md.append(f"| {e} | {n} | " + " | ".join(_f(v["per_entity"].get(e, {}).get("recall"), 2) for v in top) + " |")
        kinds = sorted({k for v in top for k in v["per_kind"]})
        md += ["", "## Leak rate per document kind (top ensembles)", "",
               "| kind | " + " | ".join(f"#{i}" for i in range(1, len(top) + 1)) + " |", "|---|" + "---|" * len(top)]
        for k in kinds:
            md.append(f"| {k} | " + " | ".join(_pct(v["per_kind"].get(k, {}).get("leak_rate")) for v in top) + " |")
        best = top[0]
        md += ["", f"## Misses and false positives of #1 ({' + '.join(best['layers'])})", ""]
        for title, key in (("Missed (gold not covered by a same-family prediction)", "missed"),
                           ("False positives (prediction overlapping no gold span)", "false_positives")):
            md.append(f"**{title}**")
            md.append("")
            for ent, d in list(best[key].items())[:12]:
                ex = ", ".join(f"`{k}`×{c}" if c > 1 else f"`{k}`" for k, c in d["examples"].items())
                md.append(f"- {ent} ({d['count']}): {ex}")
            md.append("")
    (out / "report.md").write_text("\n".join(md), encoding="utf-8")
    (out / "report.html").write_text(_html("\n".join(md), out), encoding="utf-8")


def _html(md: str, out: Path) -> str:
    """Tiny Markdown-table renderer + per-document view of the best ensemble."""
    body, table = [], []

    def flush():
        if table:
            rows = [r for r in table if not set(r.replace("|", "").strip()) <= {"-"}]
            cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
            body.append("<div class=tw><table><thead><tr>" + "".join(f"<th>{_inline(c)}</th>" for c in cells[0])
                        + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>"
                                                         for r in cells[1:]) + "</tbody></table></div>")
            table.clear()

    for line in md.splitlines():
        if line.startswith("|"):
            table.append(line)
            continue
        flush()
        if line.startswith("# "):
            body.append(f"<h1>{_inline(line[2:])}</h1>")
        elif line.startswith("## "):
            body.append(f"<h2>{_inline(line[3:])}</h2>")
        elif line.startswith("- "):
            body.append(f"<li>{_inline(line[2:])}</li>")
        elif line.strip():
            body.append(f"<p>{_inline(line)}</p>")
    flush()
    body.append(_doc_view(out))
    css = ("body{font:14px/1.45 system-ui,sans-serif;margin:24px auto;max-width:1200px;padding:0 16px;color:#1d1d1f;background:#fafafa}"
           ".tw{overflow-x:auto}table{border-collapse:collapse;margin:8px 0 20px;font-size:13px}th,td{border:1px solid #ddd;"
           "padding:4px 8px;text-align:left;vertical-align:top}th{background:#f0f0f0}code{background:#eee;padding:0 3px}"
           "mark{background:#ffe08a;border-radius:2px}mark.g{background:none;border-bottom:2px solid #2a7}"
           "mark.miss{background:#f6b0b0}pre{white-space:pre-wrap;background:#fff;border:1px solid #ddd;padding:8px}"
           "details{margin:6px 0}")
    return f"<!doctype html><meta charset=utf-8><title>PII benchmark</title><style>{css}</style>" + "\n".join(body)


def _inline(s: str) -> str:
    import re

    s = html.escape(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"`(.+?)`", r"<code>\1</code>", s)
    return re.sub(r"\*(.+?)\*", r"<i>\1</i>", s)


def _doc_view(out: Path) -> str:
    """Per document: best ensemble's predictions highlighted (yellow); for labeled
    docs, missed gold spans in red. Shows the analysed text (HTML: visible text)."""
    scores = json.loads((out / "scores.json").read_text())
    full = [v for v in scores.values() if not v.get("llm_subset") and not v["solo"] and "_preds" in v]
    if not full:
        return ""
    best = sorted(full, key=lambda v: (v["leak_rate"], -v["precision"], len(v["layers"])))[0]
    docs_path = out / "docs_view.json"
    if not docs_path.exists():
        return ""
    docs = json.loads(docs_path.read_text())
    parts = [f"<h2>Documents (best ensemble: {html.escape(' + '.join(best['layers']))})</h2>"]
    for d in docs[:300]:
        preds = best["_preds"].get(d["id"], [])
        marks = [(p["s"], p["e"], p["t"], "") for p in preds]
        marks += [(g["start"], g["end"], g["entity"], "miss") for g in d.get("gold_virtual") or []
                  if not any(p["s"] < g["end"] and g["start"] < p["e"] for p in preds)]
        marks.sort()
        t, pos, buf = d["text"], 0, []
        for a, b, ent, cls in marks:
            if a < pos:
                continue
            buf.append(html.escape(t[pos:a]))
            buf.append(f'<mark class="{cls}" title="{ent}">{html.escape(t[a:b])}</mark>')
            pos = b
        buf.append(html.escape(t[pos:]))
        parts.append(f"<details><summary>{html.escape(d['id'])} ({d['kind']}, {len(preds)} detections"
                     f"{', unlabeled' if d.get('gold_virtual') is None else ''})</summary><pre>{''.join(buf)}</pre></details>")
    return "\n".join(parts)
