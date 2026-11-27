#!/usr/bin/env python3
"""dsp_oracle backend over the Stage-3 evaluation plan (main + cf variants +
contamination probes + the alternative serialization formats).  Deterministic,
no LLM.  Writes results/raw/dsp_oracle.jsonl (overwritten each run).

The per-format oracle rows matter: they show that the magnitude / indexed /
coarse renderings carry the same extractable information as the default
triaxial one, so any model accuracy difference across formats is a pure
format effect rather than an information effect."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import evalplan  # noqa: E402
import harness  # noqa: E402


def main():
    items = evalplan.load_items(os.path.join(
        ROOT, "data", "benchmark", "items.jsonl"))
    plan = evalplan.build_plan(items)
    out = os.path.join(ROOT, "results", "raw", "dsp_oracle.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    recs = []
    for it in plan["main"]:
        r = harness.run_dsp_oracle(it)
        r["section"] = "grid"
        recs.append(r)
    for it in plan["format"]:
        for fmt in ("magnitude", "indexed", "coarse"):
            key = "serialized_" + fmt
            if key not in it:
                continue
            stub = dict(it)
            stub["serialized_signal"] = it[key]
            r = harness.run_dsp_oracle(stub)
            r["backend"] = "dsp_oracle_" + fmt
            r["section"] = "format"
            recs.append(r)
    for it in plan["cf"]:
        for v in it["counterfactual_variants"]:
            r = harness.run_dsp_oracle(it, variant=v)
            r["section"] = "cf"
            r["expected_direction"] = v.get("expected_direction")
            recs.append(r)
    for it in plan["contam"]:
        r = harness.run_dsp_oracle(it)
        r["section"] = "contam"
        recs.append(r)
    with open(out, "w") as f:
        for r in recs:
            f.write(json.dumps(r, default=str) + "\n")
    n_grid = sum(1 for r in recs if r["section"] == "grid")
    acc = sum(r["correct"] for r in recs if r["section"] == "grid") / n_grid
    print(f"wrote {len(recs)} oracle runs ({n_grid} grid, acc {acc:.3f})")


if __name__ == "__main__":
    main()
