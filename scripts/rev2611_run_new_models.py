#!/usr/bin/env python3
"""November 2026 revision runner: 2025-2026 models through Ollama.

Runs the unchanged GroundCheck conditions (and the new `code_agent`
condition) on models released after the original six, as requested in
review #63B. Output goes to results/raw_rev2611/ so the submitted paper's
raw records in results/raw/ are never modified. Resumable and append-only
like scripts/run_grid.py.

  python3 scripts/rev2611_run_new_models.py --model gpt-oss:120b \
      --think low --think-budget 2048 --tier C --sections grid
"""
import argparse
import functools
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import code_agent  # noqa: E402
import evalplan  # noqa: E402
import harness  # noqa: E402
from ollama_llm import OllamaLLM, URL as _URL  # noqa: E402
from run_grid import append, done_keys  # noqa: E402

BACKENDS = {
    "no_signal": harness.run_no_signal,
    "with_signal": harness.run_with_signal,
    "feature_prompt": harness.run_feature_prompt,
    "tool_forced": harness.run_tool_forced,
    "tool_augmented": harness.run_tool_augmented,
    "code_agent": code_agent.run_code_agent,
}
CF_BACKENDS = ["with_signal", "tool_augmented", "code_agent"]
URL_BASE = _URL.rsplit("/api/", 1)[0]


def parse_think(s):
    if s is None or s == "default":
        return None
    return {"false": False, "true": True}.get(s.lower(), s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--think", default="default",
                    help="default | false | true | low | medium | high")
    ap.add_argument("--think-budget", type=int, default=0)
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--tier", default="C", choices=["A", "B", "C"])
    ap.add_argument("--sections", nargs="+", default=["grid"])
    ap.add_argument("--backends", nargs="+", default=list(BACKENDS))
    ap.add_argument("--cf-backends", nargs="+", default=CF_BACKENDS)
    ap.add_argument("--per-family", type=int, default=None,
                    help="optional further prefix per family (probe runs)")
    ap.add_argument("--benchmark", default=os.path.join(
        ROOT, "data", "benchmark", "items.jsonl"))
    ap.add_argument("--ids", default=None,
                    help="JSON list of item ids to keep (e.g. the items whose "
                         "wording was corrected)")
    ap.add_argument("--hf", action="store_true",
                    help="--model is a HF id run with harness.HFLLM (float32, "
                         "as in the submitted runs) instead of Ollama")
    ap.add_argument("--timeout", type=int, default=900,
                    help="seconds per Ollama request (high-effort runs need "
                         "more)")
    ap.add_argument("--answer-tokens", type=int, default=None,
                    help="answer budget for no_signal/with_signal/"
                         "feature_prompt/tool_forced (default 1024 for "
                         "Ollama models, 128 for --hf as in the submission)")
    ap.add_argument("--code-tokens", type=int, default=None,
                    help="budget per code turn (default 1024 Ollama, 384 hf)")
    args = ap.parse_args()
    # Budgets (2026-11-25 12:43). Gemma 4 ignores the single-number
    # instruction and explains first, so 128 tokens cut off 150/150 of its
    # raw-text replies. New models get 1024; with greedy decoding a larger
    # cap changes only replies that would have been cut off. The original
    # six models keep the submitted budgets (--hf).
    if args.answer_tokens is None:
        args.answer_tokens = 128 if args.hf else 1024
    if args.code_tokens is None:
        args.code_tokens = 384 if args.hf else 1024
    backends = dict(BACKENDS)
    for b in ("no_signal", "with_signal", "feature_prompt", "tool_forced"):
        backends[b] = functools.partial(BACKENDS[b],
                                        max_new_tokens=args.answer_tokens)
    backends["code_agent"] = functools.partial(
        BACKENDS["code_agent"], max_new_tokens=args.code_tokens)

    items = evalplan.load_items(args.benchmark)
    plan = evalplan.build_plan(items)
    main_items = plan["main" if args.tier == "A" else "main_" + args.tier]
    if args.per_family:
        keep = []
        for fam in evalplan.FAMILIES:
            keep += [i for i in main_items if i["family"] == fam][
                :args.per_family]
        main_items = keep
    cf_items = plan["cf"] if args.tier == "A" else plan["cf_B"]
    if args.ids:
        keep_ids = set(json.load(open(args.ids)))
        main_items = [i for i in main_items if i["id"] in keep_ids]
        cf_items = [i for i in cf_items if i["id"] in keep_ids]

    think = parse_think(args.think)
    tag = args.tag or (args.model.replace(":", "-").replace("/", "-")
                       + ("" if think is None else f"-think{think}"))
    raw_dir = os.path.join(ROOT, "results", "raw_rev2611")
    os.makedirs(raw_dir, exist_ok=True)
    paths = {k: os.path.join(raw_dir, f"{k}_{tag}.jsonl")
             for k in ("grid", "cf")}
    if args.hf:
        # Same precision policy as the submitted runs (scripts/run_grid.py):
        # float32 up to 3B; bfloat16 for the 7-9B models, verified on real
        # items and falling back to float32 if long-prompt decoding degrades.
        big = any(t in args.model for t in ("7B", "8B", "9b", "9B"))
        dt = "bfloat16" if big else "float32"
        llm = harness.HFLLM(args.model, dtype=dt, stability_check=False)
        probe = []
        for fam in evalplan.FAMILIES:
            hit = next((i for i in main_items if i["family"] == fam), None)
            if hit is not None:
                probe.append(hit)
        if big:
            llm.verify_on(probe)
            if llm.stable and not llm.stable["ok"]:
                del llm
                llm = harness.HFLLM(args.model, dtype="float32",
                                    stability_check=False)
        print(f"[hf] {args.model} dtype={llm.dtype} stable={llm.stable}",
              flush=True)
        llm.meta = {}
        llm.reset_meta = lambda: None
    else:
        llm = OllamaLLM(args.model, think=think,
                        think_budget=args.think_budget, num_ctx=args.num_ctx,
                        timeout=args.timeout)
    ollama_version = None
    if not args.hf:
        try:
            import requests
            ollama_version = requests.get(
                URL_BASE + "/api/version", timeout=10).json().get("version")
        except Exception:
            pass
    cfg = {"model": args.model, "tag": tag, "think": think,
           "ollama_version": ollama_version,
           "answer_tokens": args.answer_tokens,
           "code_tokens": args.code_tokens,
           "think_budget": args.think_budget, "num_ctx": args.num_ctx,
           "tier": args.tier, "per_family": args.per_family,
           "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(raw_dir, f"config_{tag}.json"), "a") as f:
        f.write(json.dumps(cfg) + "\n")       # one line per (re)start
    t0 = time.time()

    def stamp(r):
        r.update(model=args.model, model_tag=tag, tier=args.tier,
                 ollama_version=ollama_version,
                 dtype=getattr(llm, "dtype", None),
                 answer_tokens=args.answer_tokens,
                 code_tokens=args.code_tokens,
                 device=llm.device, think=str(think),
                 think_budget=args.think_budget,
                 benchmark=os.path.relpath(args.benchmark, ROOT))
        if llm.meta:
            r["llm_meta"] = {k: v for k, v in llm.meta.items()}
        return r

    skipped = []

    def run_one(fn, it, **kw):
        # A query that times out three times is skipped, not scored: the run
        # is resumable, so a later invocation retries exactly these items.
        llm.reset_meta()
        try:
            return stamp(fn(llm, it, **kw))
        except Exception as e:  # timeouts, server restarts, and HTTP 500s
            # (Ollama's gpt-oss "harmony" parser intermittently fails with
            # 500 when the model calls its built-in python tool mid-loop)
            name = type(e).__name__
            if args.hf or not any(k in name for k in
                                  ("Timeout", "Connection", "HTTPError")):
                raise
            skipped.append(it["id"])
            print(f"[{tag}] SKIP {it['id']} ({type(e).__name__})", flush=True)
            return None

    def progress(name, k, n):
        if (k + 1) % 10 == 0 or k + 1 == n:
            print(f"[{tag}] {name} {k + 1}/{n} "
                  f"({(time.time() - t0) / 60:.1f} min)", flush=True)

    if "grid" in args.sections:
        done = done_keys(paths["grid"])
        todo = [(b, it) for b in args.backends for it in main_items
                if (b, it["id"], "base", "") not in done]
        print(f"[{tag}] grid: {len(todo)} runs", flush=True)
        for k, (b, it) in enumerate(todo):
            r = run_one(backends[b], it)
            if r is not None:
                append(paths["grid"], r)
            progress("grid", k, len(todo))

    if "cf" in args.sections:
        done = done_keys(paths["cf"])
        todo = [(b, it, v) for b in args.cf_backends for it in cf_items
                for v in it["counterfactual_variants"]
                if (b, it["id"], v["type"], "") not in done]
        print(f"[{tag}] cf: {len(todo)} runs", flush=True)
        for k, (b, it, v) in enumerate(todo):
            r = run_one(backends[b], it, variant=v)
            if r is None:
                continue
            r["expected_direction"] = v.get("expected_direction")
            append(paths["cf"], r)
            progress("cf", k, len(todo))

    if skipped:
        print(f"[{tag}] {len(skipped)} skipped (rerun to retry)", flush=True)
    print(f"[{tag}] ALL DONE in {(time.time() - t0) / 60:.1f} min",
          flush=True)


if __name__ == "__main__":
    main()
