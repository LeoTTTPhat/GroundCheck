#!/usr/bin/env python3
"""Stage-3 audit runner for ONE model (resumable, append-only JSONL).

Sections:
  grid    : main subset x {no_signal, with_signal, with_signal_cot,
            with_signal_long, feature_prompt, tool_forced, tool_augmented}
  contam  : contamination probes x with_signal, canonical arms x no_signal,
            and both arms under the PRIMED positive control
  cf      : counterfactual variants x {with_signal, tool_augmented}
  format  : format-robustness subset x {magnitude, indexed, coarse, few-shot}
  decode  : decoding-variance subset x 3 sampled decodes (T=0.7)

Every completed run is one JSON line in results/raw/<file>; reruns skip lines
already present (key: backend|item_id|variant|extra).  Safe to kill/restart.

  python3 scripts/run_grid.py --model Qwen/Qwen2.5-7B-Instruct --tier B
  python3 scripts/run_grid.py --model Qwen/Qwen2.5-1.5B-Instruct \
      --sections grid --backends with_signal tool_augmented --shard s1
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import evalplan  # noqa: E402
import harness  # noqa: E402

BACKENDS = {
    "no_signal": harness.run_no_signal,
    "with_signal": harness.run_with_signal,
    "with_signal_cot": harness.run_with_signal_cot,
    "with_signal_long": harness.run_with_signal_long,
    "feature_prompt": harness.run_feature_prompt,
    "tool_forced": harness.run_tool_forced,
    "tool_augmented": harness.run_tool_augmented,
    "tool_distractor": harness.run_tool_distractor,
}
CF_BACKENDS = ["with_signal", "tool_augmented"]
FORMATS = ["magnitude", "indexed", "coarse"]

MODEL_TAG = {
    "Qwen/Qwen2.5-0.5B-Instruct": "q0.5b",
    "Qwen/Qwen2.5-1.5B-Instruct": "q1.5b",
    "Qwen/Qwen2.5-3B-Instruct": "q3b",
    "Qwen/Qwen2.5-7B-Instruct": "q7b",
    "meta-llama/Llama-3.1-8B-Instruct": "llama8b",
    "google/gemma-2-9b-it": "gemma9b",
}


def tag_for(model):
    return MODEL_TAG.get(model, model.split("/")[-1].lower()
                         .replace("-instruct", "").replace(".", ""))


def done_keys(path):
    keys = set()
    if os.path.exists(path):
        for line in open(path):
            try:
                r = json.loads(line)
                keys.add((r["backend"], r["item_id"], r["variant"],
                          str(r.get("decode_seed", ""))))
            except (json.JSONDecodeError, KeyError):
                pass
    return keys


def append(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--full", action="store_true",
                    help="run every non-contamination item (916) instead of "
                         "the fixed subset; for split-robustness checks")
    ap.add_argument("--tier", default="A", choices=["A", "B", "C"],
                    help="A = full 500-item subset; B = nested 250-item "
                         "half; C = nested 150-item subset (slowest models)")
    ap.add_argument("--benchmark", default=os.path.join(
        ROOT, "data", "benchmark", "items.jsonl"))
    ap.add_argument("--sections", nargs="+",
                    default=["grid", "contam", "cf"])
    ap.add_argument("--backends", nargs="+", default=list(BACKENDS))
    ap.add_argument("--dtype", default="auto",
                    choices=["auto", "float32", "bfloat16", "float16"])
    ap.add_argument("--shard", default="",
                    help="suffix so parallel shards never share a file")
    args = ap.parse_args()

    items = evalplan.load_items(args.benchmark)
    plan = evalplan.build_plan(items)
    raw_dir = os.path.join(ROOT, "results", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    man_path = os.path.join(raw_dir, "eval_subset.json")
    if not os.path.exists(man_path):
        tmp = man_path + ".tmp." + str(os.getpid())
        with open(tmp, "w") as f:
            json.dump(evalplan.plan_manifest(plan), f, indent=2)
        os.rename(tmp, man_path)

    main_items = (plan["main_full"] if args.full
                  else plan["main" if args.tier == "A"
                            else "main_" + args.tier])
    cf_items = plan["cf"] if args.tier == "A" else plan["cf_B"]

    tag = tag_for(args.model)
    if args.full:
        args.shard = args.shard + "_full"
    paths = {k: os.path.join(raw_dir, f"{k}_{tag}{args.shard}.jsonl")
             for k in ("grid", "cf", "contam", "format", "decode")}

    print(f"[{tag}] loading {args.model} (tier {args.tier})", flush=True)
    llm = harness.HFLLM(args.model, dtype=args.dtype,
                        stability_check=False)
    # Probe with REAL benchmark items: a synthetic run of numbers does not
    # trigger the MPS reduced-precision decode failure even at 2.5k tokens,
    # while a real serialized item does.
    probe = []
    for fam in evalplan.FAMILIES:
        hit = next((i for i in main_items if i["family"] == fam), None)
        if hit is not None:
            probe.append(hit)
    llm.verify_on(probe)
    if llm.stable and not llm.stable["ok"] and args.dtype != "float32":
        print(f"[{tag}] {args.dtype} failed the stability probe; "
              f"falling back to float32", flush=True)
        del llm
        llm = harness.HFLLM(args.model, dtype="float32",
                            stability_check=False)
        llm.verify_on(probe)
    print(f"[{tag}] device={llm.device} dtype={llm.dtype} "
          f"sys_role={llm.supports_system} stable={llm.stable}", flush=True)
    if llm.stable and not llm.stable["ok"]:
        raise SystemExit(f"[{tag}] ABORT: long-prompt decode degenerated "
                         f"({llm.stable['probe_response']!r}). Refusing to "
                         f"record numerically invalid runs.")
    t0 = time.time()

    def stamp(r):
        r["model"] = args.model
        r["model_tag"] = tag
        r["tier"] = args.tier
        r["device"] = llm.device
        r["dtype"] = llm.dtype
        return r

    seen = {"n": 0, "deg": 0}

    def guard(rec):
        """Abort rather than record a silently-degenerating model.

        The load-time probe is necessary but not sufficient (the failure is
        prompt-specific), so every recorded response is checked as it is
        produced.
        """
        seen["n"] += 1
        seen["deg"] += bool(rec.get("degenerate"))
        if seen["n"] >= 40 and seen["deg"] / seen["n"] > 0.02:
            raise SystemExit(
                f"[{tag}] ABORT: {seen['deg']}/{seen['n']} responses "
                f"degenerated (reduced-precision decode failure). "
                f"Re-run with --dtype float32.")
        return rec

    def progress(name, k, n):
        if (k + 1) % 25 == 0 or k + 1 == n:
            print(f"[{tag}] {name} {k + 1}/{n} "
                  f"({(time.time() - t0) / 60:.1f} min)", flush=True)

    if "grid" in args.sections:
        done = done_keys(paths["grid"])
        todo = [(b, it) for b in args.backends for it in main_items
                if (b, it["id"], "base", "") not in done]
        print(f"[{tag}] grid: {len(todo)} runs", flush=True)
        for k, (b, it) in enumerate(todo):
            append(paths["grid"], guard(stamp(BACKENDS[b](llm, it))))
            progress("grid", k, len(todo))

    if "contam" in args.sections:
        done = done_keys(paths["contam"])
        probes = plan["contam"]
        canon = plan["contam_canonical"]
        pair_rank, items_by_pair = {}, {}
        for i in probes:
            pid = i["meta"]["pair_id"]
            if pid not in pair_rank:
                pair_rank[pid] = len(pair_rank)
            items_by_pair[pid] = pair_rank[pid]
        # positive control: prime with 3 verbatim canonical windows + their
        # published values, drawn from probes NOT being asked about.
        for k, it in enumerate(probes):
            if ("with_signal", it["id"], "base", "") not in done:
                append(paths["contam"],
                       guard(stamp(harness.run_with_signal(llm, it))))
            # positive control: a LOOKUP TABLE of canonical windows with the
            # queried pair's twin among them, so retrieval is arm-specific
            # (see harness.run_contam_primed). Same family as the query, so
            # the distractors are genuinely confusable.
            twin = next((c for c in canon
                         if c["meta"]["pair_id"] == it["meta"]["pair_id"]),
                        None)
            same_fam = [c for c in canon
                        if c["family"] == it["family"]
                        and c["meta"]["pair_id"] != it["meta"]["pair_id"]]
            rank = items_by_pair.get(it["meta"]["pair_id"], 0)
            distract = [same_fam[(rank + k) % len(same_fam)]
                        for k in range(3)] if same_fam else []
            table = distract[:rank % 4] + ([twin] if twin else []) \
                + distract[rank % 4:]
            primed_ids = {i["id"] for i in probes
                          if items_by_pair.get(i["meta"]["pair_id"], 0) < 30}
            if (twin is not None and it["id"] in primed_ids
                    and ("contam_primed", it["id"], "base", "") not in done):
                append(paths["contam"],
                       stamp(harness.run_contam_primed(llm, it, table)))
            progress("contam", k, len(probes))
        for it in canon:
            if ("no_signal", it["id"], "base", "") not in done:
                append(paths["contam"], stamp(harness.run_no_signal(llm, it)))
        print(f"[{tag}] contam done ({(time.time() - t0) / 60:.1f} min)",
              flush=True)

    if "cf" in args.sections:
        done = done_keys(paths["cf"])
        todo = [(b, it, v) for b in CF_BACKENDS for it in cf_items
                for v in it["counterfactual_variants"]
                if (b, it["id"], v["type"], "") not in done]
        print(f"[{tag}] cf: {len(todo)} runs", flush=True)
        for k, (b, it, v) in enumerate(todo):
            r = stamp(BACKENDS[b](llm, it, variant=v))
            r["expected_direction"] = v.get("expected_direction")
            append(paths["cf"], guard(r))
            progress("cf", k, len(todo))

    if "format" in args.sections:
        done = done_keys(paths["format"])
        fmt_items = plan["format"]
        todo = []
        for fmt in FORMATS:
            todo += [(fmt, it) for it in fmt_items
                     if ("with_signal_" + fmt, it["id"], "base", "")
                     not in done]
        todo += [("fewshot", it) for it in fmt_items
                 if ("with_signal_fewshot", it["id"], "base", "") not in done]
        print(f"[{tag}] format: {len(todo)} runs", flush=True)
        for k, (fmt, it) in enumerate(todo):
            if fmt == "fewshot":
                r = harness.run_with_signal_fewshot(llm, it)
            else:
                r = harness.run_with_signal_format(llm, it, fmt=fmt)
            append(paths["format"], stamp(r))
            progress("format", k, len(todo))

    if "decode" in args.sections:
        done = done_keys(paths["decode"])
        todo = [(s, it) for s in evalplan.DECODE_SEEDS
                for it in plan["decode"]
                if ("with_signal_sampled", it["id"], "base", str(s))
                not in done]
        print(f"[{tag}] decode: {len(todo)} runs", flush=True)
        for k, (s, it) in enumerate(todo):
            append(paths["decode"], stamp(
                harness.run_with_signal_sampled(llm, it, seed=s)))
            progress("decode", k, len(todo))

    print(f"[{tag}] ALL DONE in {(time.time() - t0) / 60:.1f} min",
          flush=True)


if __name__ == "__main__":
    main()
