#!/usr/bin/env python3
"""Does the Ollama update (0.40.17 -> 0.40.18, machine reboot 2026-11-25)
change model answers?

Reruns a fixed sample of records made under 0.40.17 with the current server
and compares predictions and correctness item by item. Greedy decoding, same
prompts, same model files; only the Ollama/llama.cpp build differs.

  GROUNDCHECK_OLLAMA=http://127.0.0.1:11441 \
      python3 scripts/rev2611_version_check.py --tag qwen3-8b-thinkFalse \
      --model qwen3:8b --think false --per-family 10

Output: results/rev2611/version_check.json (one entry per tag) and
        results/rev2611/version_check_rows.jsonl
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import evalplan  # noqa: E402
import harness  # noqa: E402
import requests  # noqa: E402
from ollama_llm import OllamaLLM, URL  # noqa: E402
from rev2611_run_new_models import parse_think  # noqa: E402

FNS = {"no_signal": harness.run_no_signal,
       "with_signal": harness.run_with_signal,
       "feature_prompt": harness.run_feature_prompt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--think", default="default")
    ap.add_argument("--think-budget", type=int, default=0)
    ap.add_argument("--per-family", type=int, default=10)
    ap.add_argument("--backends", nargs="+",
                    default=["no_signal", "with_signal"])
    args = ap.parse_args()

    version = requests.get(URL.rsplit("/api/", 1)[0] + "/api/version",
                           timeout=10).json()["version"]
    items = {i["id"]: i for i in evalplan.load_items(
        os.path.join(ROOT, "data", "benchmark", "items.jsonl"))}
    plan = evalplan.build_plan(list(items.values()))
    ids = []
    for fam in evalplan.FAMILIES:
        ids += [i["id"] for i in plan["main_C"]
                if i["family"] == fam][:args.per_family]
    old = {}
    for line in open(os.path.join(ROOT, "results", "raw_rev2611",
                                  f"grid_{args.tag}.jsonl")):
        r = json.loads(line)
        if r.get("ollama_version", "0.40.17") == "0.40.17" and \
                r["backend"] in args.backends and r["variant"] == "base":
            old[(r["backend"], r["item_id"])] = r
    llm = OllamaLLM(args.model, think=parse_think(args.think),
                    think_budget=args.think_budget)
    out_rows = os.path.join(ROOT, "results", "rev2611",
                            "version_check_rows.jsonl")
    same_pred = same_corr = n = 0
    acc_old = acc_new = 0
    for b in args.backends:
        for iid in ids:
            o = old.get((b, iid))
            if o is None:
                continue
            llm.reset_meta()
            r = FNS[b](llm, items[iid], max_new_tokens=1024)
            n += 1
            same_pred += (r["pred"] == o["pred"])
            same_corr += (r["correct"] == o["correct"])
            acc_old += o["correct"]
            acc_new += r["correct"]
            with open(out_rows, "a") as f:
                f.write(json.dumps({"tag": args.tag, "backend": b,
                                    "item_id": iid, "old_pred": o["pred"],
                                    "new_pred": r["pred"],
                                    "old_correct": o["correct"],
                                    "new_correct": r["correct"],
                                    "new_version": version}) + "\n")
    summ = {"old_version": "0.40.17", "new_version": version, "n": n,
            "identical_prediction": round(same_pred / n, 3) if n else None,
            "identical_correctness": round(same_corr / n, 3) if n else None,
            "acc_old": round(acc_old / n, 3) if n else None,
            "acc_new": round(acc_new / n, 3) if n else None,
            "backends": args.backends, "per_family": args.per_family}
    dst = os.path.join(ROOT, "results", "rev2611", "version_check.json")
    allv = json.load(open(dst)) if os.path.exists(dst) else {}
    allv[args.tag] = summ
    json.dump(allv, open(dst, "w"), indent=1)
    print(args.tag, summ)


if __name__ == "__main__":
    main()
