#!/usr/bin/env python3
"""Move records whose generation hit the token cap to superseded/, so the
resumable runner reruns exactly those with the larger budget.

With greedy decoding, raising the cap cannot change a reply that ended
before the old cap, so rerunning only the capped records gives the same
result as rerunning everything with the new budget.

  python3 scripts/rev2611_redo_capped.py --tags gemma4-26b-a4b-thinkFalse ...
"""
import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "results", "raw_rev2611")


def capped(r):
    meta = r.get("llm_meta") or {}
    return any(d == "length" for d in meta.get("done_reasons", []))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--kind", default="grid", choices=["grid", "cf"])
    ap.add_argument("--label", default="capped_v1")
    args = ap.parse_args()
    for tag in args.tags:
        p = os.path.join(RAW, f"{args.kind}_{tag}.jsonl")
        if not os.path.exists(p):
            print(tag, "no file")
            continue
        lines = open(p).readlines()
        keep = [l for l in lines if not capped(json.loads(l))]
        moved = [l for l in lines if capped(json.loads(l))]
        dst = os.path.join(RAW, "superseded",
                           f"{args.kind}_{tag}_{args.label}.jsonl")
        with open(dst, "a") as f:
            f.writelines(moved)
        with open(p + ".tmp", "w") as f:
            f.writelines(keep)
        os.replace(p + ".tmp", p)
        by = {}
        for l in moved:
            b = json.loads(l)["backend"]
            by[b] = by.get(b, 0) + 1
        print(tag, f"{len(lines)} -> {len(keep)}; moved", by)


if __name__ == "__main__":
    main()
