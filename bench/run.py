"""Benchmark runner.

    python -m bench run --data eval/out/synthetic.jsonl --out bench/out/synth
    python -m bench run --data my_texts/ --out bench/out/real --only davlan-xlmr,gliner-multi-pii
    python -m bench report --out bench/out/synth

Every candidate runs sequentially in its own subprocess (clean memory measurement,
nothing stays loaded). Raw detections are cached per layer in <out>/raw/, so
re-running only computes what is missing (--force to redo), and every ensemble
combination is scored from the cache without re-running models.

LLM candidates need a llama.cpp server: --llm-runner binary (default; --llama-server PATH
or $LLAMA_SERVER), docker (ghcr.io/ggml-org/llama.cpp:server) or external (--llm-url).
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from app.config import load_settings
from app.detection.pipeline import DetectionOptions, DetectionPipeline

from . import core
from .data import load

ROOT = Path(__file__).resolve().parent.parent


# -- helpers -------------------------------------------------------------------------------
def _settings():
    s = load_settings()
    return s


def _resources(out: Path) -> dict:
    p = out / "resources.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _save_resources(out: Path, cid: str, data: dict) -> None:
    res = _resources(out)
    res[cid] = data
    (out / "resources.json").write_text(json.dumps(res, indent=2))


def _docs(args, settings):
    docs = load(args.data, args.limit)
    if not docs:
        sys.exit("no documents found")
    return core.prepare(docs, settings)


def _pipeline(settings) -> DetectionPipeline:
    return DetectionPipeline(settings, layers=["patterns", "spacy"])


def _subprocess(cmd: list[str], env_extra: dict | None = None, python: str | None = None) -> int:
    """Run a bench stage in a fresh interpreter. `python` lets a candidate use its own
    venv (e.g. gliner2 needs transformers<5)."""
    env = {**os.environ, **(env_extra or {})}
    exe = str(ROOT / python) if python and not os.path.isabs(python) else (python or sys.executable)
    return subprocess.call([exe, "-m", "bench", *cmd], env=env, cwd=ROOT)


def _common_args(args) -> list[str]:
    out = ["--out", str(args.out), "--candidates", args.candidates]
    for d in args.data:
        out += ["--data", d]
    if args.limit:
        out += ["--limit", str(args.limit)]
    return out


# -- stage: one NER layer (subprocess) --------------------------------------------------------
def cmd_layer(args) -> None:
    settings = _settings()
    _, cands = core.load_candidates(args.candidates)
    cand = next(c for c in cands if c["id"] == args.cand)
    prepared = _docs(args, settings)
    t0 = time.perf_counter()
    layer = core.build_layer(settings, cand)
    load_s = time.perf_counter() - t0
    per_doc, t1 = {}, time.perf_counter()
    for i, p in enumerate(prepared):
        per_doc[p.doc.id] = layer.detect(p.text)
        if (i + 1) % 25 == 0:
            print(f"    {cand['id']}: {i + 1}/{len(prepared)} docs", flush=True)
    run_s = time.perf_counter() - t1
    core.write_cache(Path(args.out) / "raw" / f"{cand['id']}.jsonl", per_doc)
    chars = sum(len(p.text) for p in prepared)
    _save_resources(Path(args.out), cand["id"], {
        "kind": cand["kind"], "load_s": round(load_s, 1), "run_s": round(run_s, 1),
        "ms_per_kchar": round(run_s / max(1, chars) * 1e6, 1), "s_per_doc": round(run_s / len(prepared), 3),
        "peak_rss_gb": core.peak_rss_gb(), "docs": len(prepared)})


def write_docs_view(out: Path, prepared) -> None:
    """Analysed text + gold in analysed-text coordinates, for the report's document view."""
    rows = []
    for p in prepared:
        gold = None
        if p.doc.labeled:
            gold = []
            for g in p.doc.gold:
                if p.vt is None:
                    gold.append(g)
                    continue
                idx = [i for i, a in enumerate(p.vt.raw_start) if g["start"] <= a < g["end"]]
                if idx:
                    gold.append({"start": idx[0], "end": idx[-1] + 1, "entity": g["entity"]})
        rows.append({"id": p.doc.id, "kind": p.doc.kind, "text": p.text, "gold_virtual": gold})
    (out / "docs_view.json").write_text(json.dumps(rows, ensure_ascii=False))


# -- stage: base layers (patterns + spaCy) ------------------------------------------------------
def run_base(out: Path, prepared, settings, force: bool) -> None:
    if not force and all((out / "raw" / f"{l}.jsonl").exists() for l in core.BASE_LAYERS):
        return
    print("[base] patterns + spaCy", flush=True)
    t0 = time.perf_counter()
    pipe = _pipeline(settings)
    load_s = time.perf_counter() - t0
    caches: dict[str, dict] = {l: {} for l in core.BASE_LAYERS}
    t1 = time.perf_counter()
    for p in prepared:
        raw = pipe.collect(p.text, core.BASE_LAYERS, None)
        for l in core.BASE_LAYERS:
            caches[l][p.doc.id] = raw.get(l, [])
    run_s = time.perf_counter() - t1
    for l, per_doc in caches.items():
        core.write_cache(out / "raw" / f"{l}.jsonl", per_doc)
    _save_resources(out, "patterns+spacy", {"kind": "base", "load_s": round(load_s, 1), "run_s": round(run_s, 1),
                                            "s_per_doc": round(run_s / len(prepared), 3),
                                            "peak_rss_gb": core.peak_rss_gb(), "docs": len(prepared)})


# -- combinations -----------------------------------------------------------------------------
def finalize_combo(pipe: DetectionPipeline, prepared, caches: dict[str, dict], layers: list[str],
                   weights: dict[str, float], preset: str | None = None):
    pipe.merger.weights.update(weights)
    preds, presets = {}, {}
    for p in prepared:
        raw = {l: caches[l].get(p.doc.id, []) for l in layers}
        r = pipe.finalize(p.text, raw, DetectionOptions(preset=preset), layers)
        preds[p.doc.id] = r.spans
        presets[p.doc.id] = r.preset
    return preds, presets


def combos(ner: list[dict], max_size: int) -> list[list[str]]:
    ids = [c["id"] for c in ner]
    out = [[]]
    for k in range(1, max_size + 1):
        out += [list(x) for x in itertools.combinations(ids, k)]
    return out


# -- stage: LLM (subprocess) -------------------------------------------------------------------
def cmd_llm(args) -> None:
    from app.detection.llm_layer import LlmLayer

    from .llm_server import LlamaServer, ensure_gguf

    settings = _settings()
    defaults, cands = core.load_candidates(args.candidates)
    cand = next(c for c in cands if c["id"] == args.cand)
    d = {**defaults.get("llm", {}), **cand}
    out = Path(args.out)
    prepared = _docs(args, settings)
    prepared = prepared[: args.llm_limit] if args.llm_limit else prepared
    base_layers = args.llm_base.split(",")
    caches = {l: core.read_cache(out / "raw" / f"{l}.jsonl", prepared) for l in base_layers}
    model_path = None
    if args.llm_runner != "external":
        model_path = ensure_gguf(d["repo"], d["file"], Path(args.models_dir), args.offline)
    log_path = out / "llm" / f"{cand['id']}.server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    pipe = _pipeline(settings)
    weights = {c["id"]: c.get("weight", 0.9) for c in cands}
    pipe.merger.weights.update(weights)
    t0 = time.perf_counter()
    with LlamaServer(args.llm_runner, model_path, binary=args.llama_server, url=args.llm_url,
                     ctx_size=d.get("ctx_size", 8192), threads=d.get("threads", 6),
                     extra_args=d.get("server_args", []), log_path=log_path) as srv:
        load_s = time.perf_counter() - t0
        layer = LlmLayer(settings, base_url=srv.url, model=cand["id"],
                         disable_thinking=d.get("disable_thinking", False), extra_body=d.get("extra_body", {}),
                         max_tokens=d.get("max_tokens", settings.layers.llm.max_tokens),
                         chunk_chars=d.get("chunk_chars", settings.layers.llm.chunk_chars),
                         timeout_seconds=d.get("timeout_seconds", 1800))
        preds, warnings, t1 = {}, [], time.perf_counter()
        for i, p in enumerate(prepared):
            raw = {l: caches[l].get(p.doc.id, []) for l in base_layers}
            r = pipe.finalize(p.text, raw, DetectionOptions(), base_layers + ["llm"], use_llm=True, llm=layer)
            preds[p.doc.id] = r.spans
            warnings += r.warnings
            print(f"    {cand['id']}: {i + 1}/{len(prepared)} docs ({time.perf_counter() - t1:.0f}s)", flush=True)
        run_s = time.perf_counter() - t1
        core.write_cache(out / "llm" / f"{cand['id']}.jsonl", preds)
        _save_resources(out, cand["id"], {
            "kind": "llm", "load_s": round(load_s, 1), "run_s": round(run_s, 1),
            "s_per_doc": round(run_s / len(prepared), 1), "peak_rss_gb": srv.peak_rss_gb(),
            "docs": len(prepared), "base": base_layers, "warnings": len(warnings), **srv.throughput(),
            "gguf_gb": round(model_path.stat().st_size / 1e9, 1) if model_path else None})


# -- orchestration ----------------------------------------------------------------------------
def cmd_run(args) -> None:
    settings = _settings()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    defaults, cands = core.load_candidates(args.candidates)
    if args.only:
        wanted = set(args.only.split(","))
        cands = [c for c in cands if c["id"] in wanted or c.get("baseline")]
    if args.skip:
        cands = [c for c in cands if c["id"] not in set(args.skip.split(","))]
    prepared = _docs(args, settings)
    (out / "meta.json").write_text(json.dumps({"data": args.data, "docs": len(prepared),
                                               "labeled": sum(p.doc.labeled for p in prepared)}, indent=2))
    write_docs_view(out, prepared)
    run_base(out, prepared, settings, args.force)

    ner = [c for c in cands if c["kind"] != "llm"]
    for c in ner:
        if (out / "raw" / f"{c['id']}.jsonl").exists() and not args.force:
            continue
        print(f"[ner] {c['id']} ({c['model']})", flush=True)
        rc = _subprocess(["_layer", "--cand", c["id"], *_common_args(args)],
                         {"HF_HUB_OFFLINE": "1" if args.offline else "0"}, c.get("python"))
        if rc:
            print(f"    FAILED (exit {rc}) - skipped", flush=True)
            _save_resources(out, c["id"], {"kind": c["kind"], "error": f"exit {rc}"})

    scores = score_combos(out, prepared, settings, cands, args.max_combo)

    llms = [c for c in cands if c["kind"] == "llm"]
    if llms and not args.no_llm:
        base = args.llm_base or best_base(scores)
        print(f"[llm] base combination: {base}", flush=True)
        for c in llms:
            if (out / "llm" / f"{c['id']}.jsonl").exists() and not args.force:
                continue
            print(f"[llm] {c['id']} ({c.get('repo')}/{c.get('file')})", flush=True)
            cmd = ["_llm", "--cand", c["id"], "--llm-base", base, "--llm-runner", args.llm_runner,
                   "--models-dir", args.models_dir, "--llm-limit", str(llm_limit(c, args.llm_limit)),
                   *_common_args(args)]
            if args.llama_server:
                cmd += ["--llama-server", args.llama_server]
            if args.llm_url:
                cmd += ["--llm-url", args.llm_url]
            if args.offline:
                cmd += ["--offline"]
            rc = _subprocess(cmd)
            if rc:
                print(f"    FAILED (exit {rc}) - skipped", flush=True)
                _save_resources(out, c["id"], {"kind": "llm", "error": f"exit {rc}"})
        score_llms(out, prepared, settings, cands, llms, base, args.llm_limit)
    from .report import write_report

    write_report(out)
    print(f"report: {out / 'report.md'} and {out / 'report.html'}")


def score_combos(out: Path, prepared, settings, cands: list[dict], max_combo: int) -> dict:
    ner = [c for c in cands if c["kind"] != "llm" and (out / "raw" / f"{c['id']}.jsonl").exists()]
    caches = {l: core.read_cache(out / "raw" / f"{l}.jsonl", prepared) for l in core.BASE_LAYERS}
    for c in ner:
        caches[c["id"]] = core.read_cache(out / "raw" / f"{c['id']}.jsonl", prepared)
    weights = {c["id"]: c.get("weight", 0.9) for c in cands}
    pipe = _pipeline(settings)
    results = {}
    tried = [core.BASE_LAYERS + combo for combo in combos(ner, max_combo)]
    tried += [[c["id"]] for c in ner]  # solo: the model alone, no patterns/spaCy
    baseline = [c["id"] for c in ner if c.get("baseline")]
    print(f"[score] {len(tried)} combinations", flush=True)
    for layers in tried:
        name = "+".join(layers)
        preds, presets = finalize_combo(pipe, prepared, caches, layers, weights)
        results[name] = {"layers": layers, "solo": len(layers) == 1, "baseline": sorted(layers) == sorted(core.BASE_LAYERS + baseline),
                         **core.evaluate(prepared, preds, presets)}
        if len(prepared) <= 400:
            results[name]["_preds"] = {k: [core.span_to_json(s) for s in v] for k, v in preds.items()}
    (out / "scores.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))
    return results


def best_base(scores: dict) -> str:
    """Best non-solo combination by leak rate, ties broken by precision, then fewer layers."""
    ranked = sorted((v for v in scores.values() if not v["solo"]),
                    key=lambda v: (v["leak_rate"], -v["precision"], len(v["layers"])))
    return ",".join(ranked[0]["layers"])


def llm_limit(cand: dict, default: int) -> int:
    lim = cand.get("llm_limit")
    if lim and default:
        return min(lim, default)
    return lim or default


def score_llms(out: Path, prepared, settings, cands: list[dict], llms: list[dict], base: str, limit: int) -> None:
    """Each LLM is compared with the same base combination on the same document subset."""
    scores = json.loads((out / "scores.json").read_text())
    scores = {k: v for k, v in scores.items() if not v.get("llm_subset")}
    base_layers = base.split(",")
    weights = {c["id"]: c.get("weight", 0.9) for c in cands}
    pipe = _pipeline(settings)
    base_cache: dict[int, dict] = {}
    for c in llms:
        p = out / "llm" / f"{c['id']}.jsonl"
        if not p.exists():
            continue
        n = llm_limit(c, limit) or len(prepared)
        subset = prepared[:n]
        if n not in base_cache:
            caches = {l: core.read_cache(out / "raw" / f"{l}.jsonl", subset) for l in base_layers}
            preds, presets = finalize_combo(pipe, subset, caches, base_layers, weights)
            base_cache[n] = core.evaluate(subset, preds, presets)
        preds = core.read_cache(p, subset)
        m = core.evaluate(subset, preds)
        b = base_cache[n]
        scores[f"[llm] {c['id']}"] = {"layers": base_layers + [f"llm:{c['id']}"], "solo": False, "llm_subset": True,
                                      "llm": c["id"], "docs": n, "base": {k: b[k] for k in ("leak_rate", "recall", "precision")},
                                      **m}
    (out / "scores.json").write_text(json.dumps(scores, indent=1, ensure_ascii=False))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m bench", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--data", action="append", required=True, help="JSONL file, file or directory (repeatable)")
        p.add_argument("--out", required=True)
        p.add_argument("--candidates", default=str(ROOT / "bench" / "candidates.yaml"))
        p.add_argument("--limit", type=int, default=0, help="use only the first N documents")

    r = sub.add_parser("run", help="run candidates and write the report")
    common(r)
    r.add_argument("--only", help="comma list of candidate ids (baselines are always included)")
    r.add_argument("--skip", help="comma list of candidate ids to skip")
    r.add_argument("--force", action="store_true", help="recompute cached results")
    r.add_argument("--offline", action="store_true", help="never download models")
    r.add_argument("--max-combo", type=int, default=2, help="max number of NER models combined (default 2)")
    r.add_argument("--no-llm", action="store_true")
    r.add_argument("--llm-limit", type=int, default=16, help="documents for LLM candidates (slow on CPU); 0 = all")
    r.add_argument("--llm-base", help="layer combination the LLM runs on top of (default: best by leak rate)")
    r.add_argument("--llm-runner", choices=["binary", "docker", "external"], default="binary")
    r.add_argument("--llama-server", help="path to the llama-server binary")
    r.add_argument("--llm-url", help="external llama.cpp server (runner=external; one model at a time)")
    r.add_argument("--models-dir", default=str(ROOT / "models" / "gguf"))
    r.set_defaults(func=cmd_run)

    rep = sub.add_parser("report", help="re-render the report from cached results")
    rep.add_argument("--out", required=True)
    rep.set_defaults(func=lambda a: __import__("bench.report", fromlist=["write_report"]).write_report(Path(a.out)))

    lay = sub.add_parser("_layer")
    common(lay)
    lay.add_argument("--cand", required=True)
    lay.set_defaults(func=cmd_layer)

    llm = sub.add_parser("_llm")
    common(llm)
    for a in ("--cand", "--llm-base"):
        llm.add_argument(a, required=True)
    llm.add_argument("--llm-runner", default="binary")
    llm.add_argument("--llama-server")
    llm.add_argument("--llm-url")
    llm.add_argument("--llm-limit", type=int, default=16)
    llm.add_argument("--models-dir", default=str(ROOT / "models" / "gguf"))
    llm.add_argument("--offline", action="store_true")
    llm.set_defaults(func=cmd_llm)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
