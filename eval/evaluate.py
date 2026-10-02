"""Evaluate detection profiles against a labeled JSONL dataset.

    python -m eval.evaluate --data eval/out/synthetic.jsonl --profiles fast,balanced [--limit 50]

Metrics
  recall      gold spans overlapped by a prediction of the same family
  precision   predictions overlapping a gold span (any type)
  leak rate   share of gold PII characters NOT covered by any prediction - the
              number that matters most for redaction (type confusions don't leak)
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict

from app.config import load_settings
from app.detection.merger import family
from app.detection.pipeline import DetectionOptions, DetectionPipeline


def evaluate(pipeline: DetectionPipeline, docs: list[dict], profile: str) -> dict:
    per = defaultdict(lambda: {"gold": 0, "hit": 0})
    tp_pred = n_pred = gold_chars = leaked_chars = 0
    fps: dict[str, int] = defaultdict(int)
    t0 = time.perf_counter()
    for doc in docs:
        res = pipeline.detect(doc["text"], DetectionOptions(profile=profile))
        preds = res.spans
        covered = [False] * len(doc["text"])
        for p in preds:
            for i in range(p.start, p.end):
                covered[i] = True
        for g in doc["spans"]:
            per[g["entity"]]["gold"] += 1
            if any(p.start < g["end"] and g["start"] < p.end and family(p.entity) == family(g["entity"])
                   for p in preds):
                per[g["entity"]]["hit"] += 1
            gold_chars += g["end"] - g["start"]
            leaked_chars += sum(1 for i in range(g["start"], g["end"])
                                if not covered[i] and not doc["text"][i].isspace())
        for p in preds:
            n_pred += 1
            if any(p.start < g["end"] and g["start"] < p.end for g in doc["spans"]):
                tp_pred += 1
            else:
                fps[p.entity] += 1
    gold_total = sum(v["gold"] for v in per.values())
    hit_total = sum(v["hit"] for v in per.values())
    return {
        "profile": profile,
        "docs": len(docs),
        "seconds_per_doc": round((time.perf_counter() - t0) / max(1, len(docs)), 3),
        "recall": round(hit_total / max(1, gold_total), 4),
        "precision": round(tp_pred / max(1, n_pred), 4),
        "leak_rate": round(leaked_chars / max(1, gold_chars), 4),
        "per_entity_recall": {k: round(v["hit"] / v["gold"], 3) for k, v in sorted(per.items())},
        "false_positives_by_type": dict(sorted(fps.items(), key=lambda x: -x[1])),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="eval/out/synthetic.jsonl")
    ap.add_argument("--profiles", default="fast,balanced")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--llm-url", default=None)
    ap.add_argument("--out", default=None, help="write JSON report here")
    args = ap.parse_args()
    settings = load_settings()
    if args.llm_url:
        settings.layers.llm.base_url = args.llm_url
    docs = [json.loads(line) for line in open(args.data, encoding="utf-8")]
    if args.limit:
        docs = docs[: args.limit]
    profiles = args.profiles.split(",")
    layers = sorted({l for p in profiles for l in settings.profiles[p]})
    pipeline = DetectionPipeline(settings, layers)
    reports = [evaluate(pipeline, docs, p) for p in profiles]
    for r in reports:
        print(f"\n== {r['profile']}: recall {r['recall']:.3f}  precision {r['precision']:.3f}  "
              f"leak rate {r['leak_rate']:.3%}  ({r['seconds_per_doc']} s/doc, {r['docs']} docs)")
        for ent, rec in r["per_entity_recall"].items():
            print(f"   {ent:18} {rec:.3f}")
        print("   false positives:", r["false_positives_by_type"])
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(reports, fh, indent=2)


if __name__ == "__main__":
    main()
