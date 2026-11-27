#!/usr/bin/env python3
"""Validate the dtype choice for the large models.

The small Qwen models must run in float32 (bf16 degenerates on long numeric
prompts -- see src/test_stability_probe.py).  The 7-9B models are run in
bfloat16 because float32 roughly doubles their cost, guarded by the
per-response degeneracy check.  That is only defensible if bf16 and float32
actually produce the same answers, so this script measures it directly:
one model, one item subset, both dtypes, greedy decoding.

Writes results/dtype_agreement.json.

  python3 scripts/check_dtype_agreement.py --model Qwen/Qwen2.5-7B-Instruct --n 50
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import evalplan  # noqa: E402
import harness  # noqa: E402
import stats as st  # noqa: E402

BACKENDS = {"no_signal": harness.run_no_signal,
            "with_signal": harness.run_with_signal,
            "tool_augmented": harness.run_tool_augmented}


def run(model, dtype, items, backends):
    import torch
    llm = harness.HFLLM(model, dtype=dtype, stability_check=False)
    llm.verify_on(items[:5])
    out = {}
    for b in backends:
        for it in items:
            r = BACKENDS[b](llm, it)
            out[(b, it["id"])] = (r["pred"], r["correct"], r["degenerate"])
    del llm
    torch.mps.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--backends", nargs="+",
                    default=["no_signal", "with_signal", "tool_augmented"])
    args = ap.parse_args()

    all_items = evalplan.load_items(
        os.path.join(ROOT, "data", "benchmark", "items.jsonl"))
    plan = evalplan.build_plan(all_items)
    per_fam = max(1, args.n // len(evalplan.FAMILIES))
    items = []
    for fam in evalplan.FAMILIES:
        items += [i for i in plan["main_C"] if i["family"] == fam][:per_fam]

    a = run(args.model, "bfloat16", items, args.backends)
    b = run(args.model, "float32", items, args.backends)

    keys = sorted(set(a) & set(b), key=str)
    same_pred = [int(str(a[k][0]) == str(b[k][0])) for k in keys]
    ca = [a[k][1] for k in keys]
    cb = [b[k][1] for k in keys]
    d, lo, hi = st.boot_diff_ci(ca, cb)
    mc = st.mcnemar(ca, cb)
    res = {
        "model": args.model, "n_items": len(items), "n_queries": len(keys),
        "backends": args.backends,
        "identical_answer_rate": round(sum(same_pred) / len(keys), 4),
        "acc_bfloat16": round(sum(ca) / len(ca), 4),
        "acc_float32": round(sum(cb) / len(cb), 4),
        "acc_diff": round(d, 4), "acc_diff_ci": [round(lo, 4), round(hi, 4)],
        "mcnemar_p": round(mc["p"], 4),
        "equivalent_at_0.05": st.tost_equivalence(ca, cb,
                                                  margin=0.05)["equivalent"],
        "degenerate_bfloat16": sum(v[2] for v in a.values()),
        "degenerate_float32": sum(v[2] for v in b.values()),
        "per_backend": {},
    }
    for bk in args.backends:
        ks = [k for k in keys if k[0] == bk]
        if ks:
            res["per_backend"][bk] = {
                "identical_answer_rate": round(
                    sum(int(str(a[k][0]) == str(b[k][0])) for k in ks)
                    / len(ks), 4),
                "acc_bfloat16": round(sum(a[k][1] for k in ks) / len(ks), 4),
                "acc_float32": round(sum(b[k][1] for k in ks) / len(ks), 4),
                "n": len(ks)}
    out = os.path.join(ROOT, "results", "dtype_agreement.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
