#!/usr/bin/env python3
"""GroundCheck smoke test.

1. Runs the ground-truth engine unit tests (synthetic oracle validation).
2. Validates dsp_oracle agreement with stored ground truth on the full
   benchmark (serialization-fidelity ceiling).
3. Runs 5 items (one per family) through EVERY backend:
   dsp_oracle, and {no_signal, with_signal, tool_augmented} x
   {Qwen2.5-0.5B-Instruct, Qwen2.5-1.5B-Instruct} on MPS.

Writes results/smoke_results.json and results/SMOKE.md.
Usage: python3 scripts/run_smoke.py [--n_per_family 1] [--models ...]
"""
import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import harness  # noqa: E402

FAMILIES = ["peak_count", "dominant_freq", "intensity_order",
            "transition_time", "bout_duration"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_per_family", type=int, default=1)
    ap.add_argument("--models", nargs="+",
                    default=["Qwen/Qwen2.5-0.5B-Instruct",
                             "Qwen/Qwen2.5-1.5B-Instruct"])
    ap.add_argument("--benchmark", default=os.path.join(
        ROOT, "data", "benchmark", "items.jsonl"))
    args = ap.parse_args()

    report = {"config": vars(args), "timestamp": time.strftime("%F %T")}

    # 1. oracle unit tests -------------------------------------------------
    p = subprocess.run([sys.executable,
                        os.path.join(ROOT, "src", "test_ground_truth.py")],
                       capture_output=True, text=True)
    report["oracle_unit_tests"] = {"returncode": p.returncode,
                                   "output": p.stdout.strip().splitlines()}
    print(p.stdout.strip())
    assert p.returncode == 0, "oracle unit tests failed"

    # 2. dsp_oracle vs stored ground truth on the whole benchmark ----------
    items = [json.loads(l) for l in open(args.benchmark)]
    agree = defaultdict(list)
    for it in items:
        r = harness.run_dsp_oracle(it)
        agree[it["family"]].append(r["correct"])
    report["dsp_oracle_vs_groundtruth"] = {
        f: {"acc": round(sum(v) / len(v), 4), "n": len(v)}
        for f, v in agree.items()}
    print("dsp_oracle agreement:", report["dsp_oracle_vs_groundtruth"])

    # 3. pick smoke items: first item of each family (deterministic) -------
    smoke_items = []
    for fam in FAMILIES:
        fam_items = [i for i in items if i["family"] == fam and
                     not i["meta"].get("contamination_probe")]
        smoke_items.extend(fam_items[:args.n_per_family])
    report["smoke_item_ids"] = [i["id"] for i in smoke_items]

    results = []
    for it in smoke_items:
        results.append(harness.run_dsp_oracle(it))

    for model_name in args.models:
        print(f"-- loading {model_name}")
        llm = harness.HFLLM(model_name)
        for it in smoke_items:
            for fn in (harness.run_no_signal, harness.run_with_signal,
                       harness.run_tool_augmented):
                r = fn(llm, it)
                r["model"] = model_name
                r["device"] = llm.device
                results.append(r)
                print(f"{model_name.split('/')[-1]:28s} {r['backend']:15s} "
                      f"{it['family']:16s} pred={r['pred']} "
                      f"gt={r['ground_truth']} ok={r['correct']} "
                      f"{r['latency_s']:.1f}s")
        del llm

    report["results"] = results

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "smoke_results.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)

    write_md(report)
    print("wrote results/SMOKE.md")


def write_md(report):
    lines = ["# GroundCheck smoke test", "",
             f"Run: {report['timestamp']}  ",
             f"Benchmark: {report['config']['benchmark']}", "",
             "## 1. Ground-truth engine unit tests (synthetic signals)", ""]
    lines += ["```"] + report["oracle_unit_tests"]["output"] + ["```", ""]
    lines += ["## 2. dsp_oracle vs stored ground truth (full benchmark)",
              "",
              "Fraction of items where the DSP pipeline applied to the "
              "SERIALIZED (10 Hz, rounded) signal reproduces the full-rate "
              "programmatic ground truth within tolerance — the ceiling any "
              "prompted model can reach:", "",
              "| family | acc | n |", "|---|---|---|"]
    for f, v in report["dsp_oracle_vs_groundtruth"].items():
        lines.append(f"| {f} | {v['acc']} | {v['n']} |")
    lines += ["", "## 3. Five items through every backend", "",
              "| model | backend | family | pred | ground truth | correct | "
              "prompt tok | gen tok | latency s |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in report["results"]:
        model = r.get("model", "-").split("/")[-1]
        lines.append(
            f"| {model} | {r['backend']} | {r['family']} | {r['pred']} | "
            f"{r['ground_truth']} | {r['correct']} | {r['prompt_tokens']} | "
            f"{r['gen_tokens']} | {r['latency_s']} |")
    n_llm = [r for r in report["results"] if r["backend"] != "dsp_oracle"]
    n_ok = sum(1 for r in n_llm if r["pred"] is not None)
    lines += ["",
              f"LLM runs: {len(n_llm)}, parseable answers: {n_ok}. "
              f"Device: MPS (see results/smoke_results.json raw output).", ""]
    with open(os.path.join(ROOT, "results", "SMOKE.md"), "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
