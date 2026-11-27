#!/usr/bin/env python3
"""Aggregate Stage-3 raw runs into results/metrics.json + results/RESULTS.md.

Reads results/raw/{grid,cf,contam,format,decode}_*.jsonl + dsp_oracle.jsonl.
Every number is computed from those files; nothing is entered by hand.

Stage-3 additions over Stage 2:
  * Wilson 95% CIs on every rate; item-level bootstrap CIs on every gap.
  * Exact paired McNemar tests for with_signal vs no_signal (and for
    tool_augmented vs tool_forced), Holm-corrected across families, plus a
    TOST equivalence test so 'the signal adds nothing' is an established
    claim rather than a failure to reject.
  * Six models (0.5B-9B, three families) with a Spearman scale-trend test.
  * transition_time split by provenance (real PAMAP2 consensus vs spliced).
  * Routing-vs-reading decomposition (tool_augmented / tool_forced /
    feature_prompt).
  * Response-diversity statistics separating degeneracy from ungroundedness.
  * Format-robustness and decoding-variance sections.
  * Contamination positive control (in-context primed memoriser).
"""
import glob
import json
import os
import sys
import time
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import evalplan  # noqa: E402
import harness  # noqa: E402
import stats as st  # noqa: E402

RAW = os.path.join(ROOT, "results", "raw")
FAMILIES = evalplan.FAMILIES
BACKENDS = ["no_signal", "with_signal", "with_signal_cot",
            "with_signal_long", "feature_prompt", "tool_forced",
            "tool_augmented", "tool_distractor"]
SIG_BACKENDS = ["with_signal", "tool_augmented"]
FORMAT_BACKENDS = ["with_signal_magnitude", "with_signal_indexed",
                   "with_signal_coarse", "with_signal_fewshot"]

# ordered small -> large; params in billions drive the scale-trend test
MODELS = [
    ("Qwen/Qwen2.5-0.5B-Instruct", "Qwen2.5-0.5B", 0.5, "Qwen2.5"),
    ("Qwen/Qwen2.5-1.5B-Instruct", "Qwen2.5-1.5B", 1.5, "Qwen2.5"),
    ("Qwen/Qwen2.5-3B-Instruct", "Qwen2.5-3B", 3.0, "Qwen2.5"),
    ("Qwen/Qwen2.5-7B-Instruct", "Qwen2.5-7B", 7.0, "Qwen2.5"),
    ("meta-llama/Llama-3.1-8B-Instruct", "Llama-3.1-8B", 8.0, "Llama-3.1"),
    ("google/gemma-2-9b-it", "Gemma-2-9B", 9.0, "Gemma-2"),
]
MODEL_IDS = [m[0] for m in MODELS]
SHORT = {m[0]: m[1] for m in MODELS}
SIZE = {m[0]: m[2] for m in MODELS}
EQUIV_MARGIN = 0.05     # TOST margin: |gap| < 5 points == 'no useful gain'


def load(pattern):
    out, seen = [], set()
    for p in sorted(glob.glob(os.path.join(RAW, pattern))):
        for line in open(p):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:      # torn tail line after a kill
                continue
            key = (r.get("model"), r["backend"], r["item_id"], r["variant"],
                   r.get("decode_seed"))
            if key in seen:
                continue
            seen.add(key)
            # Recompute the degeneracy flag from the stored text rather than
            # trusting the value written at run time, so a detector fix
            # applies retroactively to every raw record already on disk.
            r["degenerate"] = harness.is_degenerate(r.get("raw_response"))
            out.append(r)
    return out


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def band(tol, ref):
    if tol["type"] == "absolute":
        return tol["value"]
    if tol["type"] == "relative":
        r = num(ref)
        return max(tol["value"] * abs(r if r is not None else 0),
                   tol.get("min_abs", 0.0))
    return None


def unchanged(item, base_pred, var_pred):
    tol = item["tolerance"]
    if tol["type"] == "exact":
        return str(var_pred).upper() == str(base_pred).upper()
    b, v = num(base_pred), num(var_pred)
    if b is None or v is None:
        return None
    return abs(v - b) <= band(tol, b)


def r3(x):
    return None if x is None else round(x, 4)


def prov(item):
    """transition_time provenance: real (PAMAP2 consensus) vs spliced."""
    if item["family"] != "transition_time":
        return None
    return "real" if item["meta"].get("real_transition") else "spliced"


def ci_entry(flags):
    p, lo, hi = st.wilson_from_flags(flags)
    return {"acc": r3(p), "ci_lo": r3(lo), "ci_hi": r3(hi),
            "n": len([f for f in flags if f is not None])}


def main():
    all_items = evalplan.load_items(
        os.path.join(ROOT, "data", "benchmark", "items.jsonl"))
    items = {i["id"]: i for i in all_items}
    plan = evalplan.build_plan(all_items)

    # The append-only raw files can retain rows from an earlier, larger
    # evaluation plan.  Restrict every section to the CURRENT plan's item
    # ids, otherwise models would silently be scored on different item sets
    # and the paired comparisons would no longer be paired.
    def keep(rows, section):
        ok = {i["id"] for i in plan[section]}
        return [r for r in rows if r["item_id"] in ok]

    grid_all = load("grid_*.jsonl")
    grid = keep(grid_all, "main")
    cf = keep(load("cf_*.jsonl"), "cf")
    contam = keep(load("contam_*.jsonl"), "contam")
    fmt = keep(load("format_*.jsonl"), "format")
    dec = keep(load("decode_*.jsonl"), "decode")
    oracle = load("dsp_oracle.jsonl")
    oracle = [r for r in oracle
              if r["item_id"] in {i["id"] for i in
                                  plan["main"] + plan["cf"] + plan["contam"]
                                  + plan["format"]}]

    present = [m for m in MODEL_IDS
               if any(r["model"] == m for r in grid)]
    # the full-pool runs live in the same grid files but outside the shared
    # subset; they back a reported analysis (Sec. V-B), so they count toward
    # the measured-query total even though they are excluded from the
    # paired main grid
    _sub = {i["id"] for i in plan["main"]}
    _full = {i["id"] for i in plan.get("main_full", [])}
    grid_extra = [r for r in grid_all
                  if r["item_id"] in _full and r["item_id"] not in _sub]
    M = {"timestamp": time.strftime("%F %T"),
         "models_present": [SHORT[m] for m in present],
         "n_raw_runs": {"grid": len(grid), "grid_full_pool": len(grid_extra),
                        "cf": len(cf),
                        "contam": len(contam), "format": len(fmt),
                        "decode": len(dec), "dsp_oracle": len(oracle)},
         "n_raw_total": len(grid) + len(grid_extra) + len(cf) + len(contam)
         + len(fmt) + len(dec) + len(oracle),
         "equivalence_margin": EQUIV_MARGIN,
         "degenerate_responses": {
             "total": sum(bool(r.get("degenerate"))
                          for r in grid + grid_extra + cf + contam + fmt
                          + dec),
             "by_model": {SHORT.get(m, m): sum(
                 bool(r.get("degenerate"))
                 for r in grid + grid_extra + cf + contam + fmt + dec
                 if r.get("model") == m) for m in MODEL_IDS},
             "note": "recomputed from raw_response by harness.is_degenerate; "
                     "a nonzero count invalidates the affected model's runs"}}

    # -------------------------------------------- best-constant baseline --
    # A tolerance band plus a narrow ground-truth distribution means a single
    # constant answer can score well without reading anything: Qwen2.5-7B
    # answers "24" on every peak_count item and lands inside the +-20% band
    # 85% of the time.  Accuracy is therefore uninterpretable without the
    # best score any constant could achieve, so we compute it exactly (the
    # optimal constant is always one of the ground-truth values, since the
    # covered set only changes as the constant crosses a band edge).
    def _const_acc(c, pool):
        """Hits of answering the constant c on the items in pool.

        Each item is scored with its OWN tolerance. (Rev. 2026-11: the
        submitted version applied the first item's tolerance to the whole
        family, which scored spliced transitions at +-2 s instead of their
        +-1 s and raised the deployable floor from 0.580 to 0.613.)
        """
        hits = 0
        for i in pool:
            g, tol = i["ground_truth"], i["tolerance"]
            if tol["type"] == "exact":
                hits += str(g).upper() == str(c).upper()
            elif num(g) is not None:
                hits += abs(c - num(g)) <= band(tol, g)
        return hits

    def best_constant(pool, fit_pool=None):
        """Best single answer for a family.

        R1/R2: the constant chosen on the evaluation items themselves is an
        *oracle* quantity -- it sees the labels, like reporting the best
        majority class after inspecting the test set.  It is the right
        diagnostic for benchmark concentration but is not a baseline a
        practitioner could deploy.  Passing `fit_pool` selects the constant on
        held-out items and scores it on `pool`, which is the deployable
        version.  We report both.
        """
        if not pool:
            return None
        tol = pool[0]["tolerance"]
        gts = [i["ground_truth"] for i in pool]
        cands = [i["ground_truth"] for i in (fit_pool if fit_pool else pool)]
        if tol["type"] == "exact":
            cands = [str(c).upper() for c in cands]
            gts_s = [str(g).upper() for g in gts]
            best = max(set(cands), key=cands.count) if cands else None
            hit = sum(1 for g in gts_s if g == best)
            return {"acc": r3(hit / len(gts_s)), "answer": best,
                    "n": len(gts_s)}
        cands = sorted({num(c) for c in cands if num(c) is not None})
        if not cands or not pool:
            return None
        fit = fit_pool if fit_pool else pool
        best_c = max(cands, key=lambda c: _const_acc(c, fit))
        return {"acc": r3(_const_acc(best_c, pool) / len(pool)),
                "answer": best_c, "n": len(pool)}

    # oracle constant (fit on the evaluated items) and a deployable constant
    # (fit on the held-out remainder of the 1,036-item pool, scored on the
    # evaluated items).
    evaluated = plan["main"]
    heldout = [i for i in all_items
               if not i["meta"].get("contamination_probe")
               and i["id"] not in {j["id"] for j in evaluated}]
    M["best_constant"] = {"per_family": {}, "deployable_per_family": {}}
    for f in FAMILIES:
        ev = [i for i in evaluated if i["family"] == f]
        ho = [i for i in heldout if i["family"] == f]
        M["best_constant"]["per_family"][f] = best_constant(ev)
        M["best_constant"]["deployable_per_family"][f] = (
            best_constant(ev, fit_pool=ho) if ho else None)
    def _pool_overall(key):
        tot = n_tot = 0
        for f in FAMILIES:
            e = M["best_constant"][key].get(f)
            if e:
                tot += e["acc"] * e["n"]
                n_tot += e["n"]
        return r3(tot / n_tot) if n_tot else None
    M["best_constant"]["overall"] = _pool_overall("per_family")
    M["best_constant"]["deployable_overall"] = _pool_overall(
        "deployable_per_family")
    M["best_constant"]["n_heldout"] = len(heldout)

    # ---------------------------------------- benchmark concentration ----
    # R2: a high floor means the truths are concentrated relative to the
    # tolerance band.  Quantify that directly rather than leaving it implied.
    import math
    M["concentration"] = {}
    for f in FAMILIES:
        ev = [i for i in evaluated if i["family"] == f]
        if not ev:
            continue
        tol = ev[0]["tolerance"]
        gts = [i["ground_truth"] for i in ev]
        if tol["type"] == "exact":
            vals = [str(g).upper() for g in gts]
            counts = {v: vals.count(v) for v in set(vals)}
            bins = len(counts)
        else:
            nums = sorted(num(g) for g in gts if num(g) is not None)
            # greedy cover: how many disjoint tolerance bands span the
            # truths, and how the truths distribute over those bands (the
            # entropy of a numeric family is only defined once the
            # tolerance has quantized it)
            bins, i, counts = 0, 0, {}
            while i < len(nums):
                edge = nums[i] + 2 * band(tol, nums[i])
                k = 0
                while i < len(nums) and nums[i] <= edge:
                    i += 1
                    k += 1
                counts[bins] = k
                bins += 1
        ps = [c / len(gts) for c in counts.values()] if counts else None
        ent = (-sum(p * math.log(p) for p in ps) if ps else None)
        M["concentration"][f] = {
            "n": len(gts), "distinguishable_bins": bins,
            "entropy_nats": r3(ent) if ent is not None else None,
            "floor": M["best_constant"]["per_family"][f]["acc"]}

    # ------------------------------------- full-benchmark generalization ---
    # Two models were additionally run over the whole 916-item
    # non-contamination pool.  The 300-item main subset is what every model
    # shares, so the full pool is reported as a separate check that the
    # subset is representative rather than folded into the main grid.
    full_ids = {i["id"] for i in plan.get("main_full", [])}
    sub_ids = {i["id"] for i in plan["main"]}
    M["full_benchmark"] = {}
    for m in MODEL_IDS:
        rows = [r for r in grid_all
                if r["model"] == m and r["item_id"] in full_ids]
        # a full-pool run is one that reaches items outside the shared
        # subset; every model has subset rows, so count only the extras
        extra = {r["item_id"] for r in rows} - sub_ids
        if len(extra) < 0.5 * (len(full_ids) - len(sub_ids)):
            continue
        ent = {}
        for b in BACKENDS:
            br = [r for r in rows if r["backend"] == b]
            if not br:
                continue
            sub = [r["correct"] for r in br if r["item_id"] in sub_ids]
            if not sub or len(br) < 0.9 * len(full_ids):
                continue
            ent[b] = {
                "full_acc": r3(sum(r["correct"] for r in br) / len(br)),
                "n_full": len(br),
                "subset_acc": r3(sum(sub) / len(sub)),
                "n_subset": len(sub),
                "delta": r3(sum(r["correct"] for r in br) / len(br)
                            - sum(sub) / len(sub))}
        if ent:
            M["full_benchmark"][SHORT.get(m, m)] = ent

    # ---------------------------------------------------- coverage counts ---
    M["coverage"] = {
        "n_items": len(items),
        "n_main": len(plan["main"]),
        "n_main_full": len(plan.get("main_full", [])),
        "n_cf": len(plan.get("cf", [])),
        "n_contam": len(plan.get("contam", [])),
    }

    # ------------------------------------------------------ dsp_oracle ---
    og = [r for r in oracle if r["section"] == "grid"]
    M["dsp_oracle"] = {"per_family": {}, "per_provenance": {}}
    for f in FAMILIES:
        M["dsp_oracle"]["per_family"][f] = ci_entry(
            [r["correct"] for r in og if r["family"] == f])
    for pv in ("real", "spliced"):
        M["dsp_oracle"]["per_provenance"][pv] = ci_entry(
            [r["correct"] for r in og if prov(items[r["item_id"]]) == pv])
    M["dsp_oracle"]["overall"] = ci_entry([r["correct"] for r in og])
    of = [r for r in oracle if r.get("section") == "format"]
    M["dsp_oracle"]["per_format"] = {
        b.replace("dsp_oracle_", ""): ci_entry(
            [r["correct"] for r in of if r["backend"] == b])
        for b in sorted({r["backend"] for r in of})}

    # ------------------------------------------------------------ grid ---
    # per-item correctness maps, so every comparison below is PAIRED on ids
    corr = defaultdict(dict)      # (model, backend) -> {item_id: 0/1}
    pred = defaultdict(dict)
    for r in grid:
        corr[(r["model"], r["backend"])][r["item_id"]] = r["correct"]
        pred[(r["model"], r["backend"])][r["item_id"]] = r["pred"]
    for r in og:
        corr[("dsp_oracle", "dsp_oracle")][r["item_id"]] = r["correct"]
        pred[("dsp_oracle", "dsp_oracle")][r["item_id"]] = r["pred"]

    M["grid"] = {}
    for m in present:
        M["grid"][m] = {"short": SHORT[m], "size_b": SIZE[m]}
        for b in BACKENDS:
            runs = [r for r in grid if r["model"] == m and r["backend"] == b]
            if not runs:
                continue
            ent = {"per_family": {}, "per_provenance": {},
                   "diversity": {}, "n": len(runs)}
            for f in FAMILIES:
                fr = [r for r in runs if r["family"] == f]
                if not fr:
                    continue
                ent["per_family"][f] = ci_entry([r["correct"] for r in fr])
                ent["per_family"][f]["parse_rate"] = r3(
                    sum(r["pred"] is not None for r in fr) / len(fr))
                ent["diversity"][f] = {
                    k: r3(v) for k, v in st.response_diversity(
                        [r["pred"] for r in fr]).items()}
            for pv in ("real", "spliced"):
                pr = [r for r in runs if prov(items[r["item_id"]]) == pv]
                if pr:
                    ent["per_provenance"][pv] = ci_entry(
                        [r["correct"] for r in pr])
            ent["overall"] = ci_entry([r["correct"] for r in runs])
            fl = M["best_constant"]["overall"]
            ce = M["dsp_oracle"]["overall"]["acc"]
            if fl is not None and ce is not None and ce > fl:
                ent["above_floor"] = r3(
                    (ent["overall"]["acc"] - fl) / (ce - fl))
            ent["parse_rate"] = r3(sum(r["pred"] is not None for r in runs)
                                   / len(runs))
            ent["mean_latency_s"] = r3(sum(r["latency_s"] for r in runs)
                                       / len(runs))
            ent["mean_prompt_tokens"] = r3(sum(r["prompt_tokens"]
                                               for r in runs) / len(runs))
            ent["mean_gen_tokens"] = r3(sum(r["gen_tokens"] for r in runs)
                                        / len(runs))
            ent["gen_cap_hit_rate"] = r3(sum(
                r["gen_tokens"] >= {"no_signal": 128, "with_signal": 128,
                                    "with_signal_cot": 192,
                                    "with_signal_long": 256,
                                    "feature_prompt": 128,
                                    "tool_forced": 128,
                                    "tool_augmented": 96,
                                    "tool_distractor": 96}[b]
                for r in runs) / len(runs))
            M["grid"][m][b] = ent

    # -------------------------------------------- tolerance sensitivity --
    # R2: are the floor, the ceiling and the ordering artifacts of the chosen
    # bands?  Re-score the oracle and the constant floor under tighter and
    # looser tolerances.
    def _scaled_tol(tol, k):
        t = dict(tol)
        if t["type"] != "exact":
            t["value"] = tol["value"] * k
            if "min_abs" in t:
                t["min_abs"] = tol["min_abs"] * k
        return t

    oracle_pred = {r["item_id"]: r["pred"] for r in og}
    M["tolerance_sweep"] = {}
    for k in (0.5, 1.0, 2.0):
        fl = cl = n_all = 0
        for f in FAMILIES:
            ev = [i for i in evaluated if i["family"] == f]
            if not ev:
                continue
            tols = {i["id"]: _scaled_tol(i["tolerance"], k) for i in ev}
            if ev[0]["tolerance"]["type"] == "exact":
                best = max({str(i["ground_truth"]).upper() for i in ev},
                           key=lambda c: sum(
                               1 for i in ev
                               if str(i["ground_truth"]).upper() == c))
                fl += sum(1 for i in ev
                          if str(i["ground_truth"]).upper() == best)
                cl += sum(1 for i in ev
                          if str(oracle_pred.get(i["id"])).upper()
                          == str(i["ground_truth"]).upper())
            else:
                evn = [i for i in ev if num(i["ground_truth"]) is not None]
                cands = sorted({num(i["ground_truth"]) for i in evn})
                fl += max(sum(1 for i in evn
                              if abs(c - num(i["ground_truth"]))
                              <= band(tols[i["id"]], i["ground_truth"]))
                          for c in cands) if cands else 0
                for i in ev:
                    pv, gv = num(oracle_pred.get(i["id"])), \
                        num(i["ground_truth"])
                    if pv is not None and gv is not None and \
                            abs(pv - gv) <= band(tols[i["id"]], gv):
                        cl += 1
            n_all += len(ev)
        M["tolerance_sweep"][f"x{k:g}"] = {
            "floor": r3(fl / n_all), "ceiling": r3(cl / n_all),
            "headroom": r3((cl - fl) / n_all)}

    # ------------------------------------------------ split robustness ---
    # R2: item-level bootstrap conditions on the realized 300-item subset.
    # Recompute the headline gap on disjoint halves to expose split variance.
    M["split_robustness"] = {}
    ev_ids = [i["id"] for i in evaluated]
    rng = __import__("numpy").random.default_rng(7)
    for m in present:
        base = corr.get((m, "no_signal"), {})
        cur = corr.get((m, "with_signal"), {})
        ids = [i for i in ev_ids if i in base and i in cur]
        if len(ids) < 60:
            continue
        gaps = []
        for rep in range(20):
            perm = rng.permutation(len(ids))
            for half in (perm[:len(ids) // 2], perm[len(ids) // 2:]):
                sub = [ids[j] for j in half]
                gaps.append(sum(cur[i] - base[i] for i in sub) / len(sub))
        M["split_robustness"][SHORT[m]] = {
            "gap_full": r3(sum(cur[i] - base[i] for i in ids) / len(ids)),
            "half_split_min": r3(min(gaps)), "half_split_max": r3(max(gaps)),
            "half_split_sd": r3(float(__import__("numpy").std(gaps))),
            "n_halves": len(gaps)}

    # ------------------------------- prior-reliance gap + paired tests ---
    M["prior_reliance_gap"] = {}
    for m in present:
        if (m, "no_signal") not in corr:
            continue
        base = corr[(m, "no_signal")]
        M["prior_reliance_gap"][m] = {}
        for b in BACKENDS:
            if b == "no_signal" or (m, b) not in corr:
                continue
            cur = corr[(m, b)]
            ids = sorted(set(base) & set(cur))
            per, pv_list, pv_keys = {}, [], []
            for f in FAMILIES:
                fid = [i for i in ids if items[i]["family"] == f]
                if not fid:
                    continue
                a = [cur[i] for i in fid]
                z = [base[i] for i in fid]
                d, lo, hi = st.boot_diff_ci(a, z)
                mc = st.mcnemar(a, z)
                per[f] = {"gap": r3(d), "ci_lo": r3(lo), "ci_hi": r3(hi),
                          "p": r3(mc["p"]), "n": len(fid),
                          "n_better": mc["n10"], "n_worse": mc["n01"]}
                pv_list.append(mc["p"])
                pv_keys.append(f)
            adj, rej = st.holm(pv_list)
            for k, f in enumerate(pv_keys):
                per[f]["p_holm"] = r3(float(adj[k]))
                per[f]["significant"] = bool(rej[k])
            a = [cur[i] for i in ids]
            z = [base[i] for i in ids]
            d, lo, hi = st.boot_diff_ci(a, z)
            mc = st.mcnemar(a, z)
            tost = st.tost_equivalence(a, z, margin=EQUIV_MARGIN)
            per["overall"] = {"gap": r3(d), "ci_lo": r3(lo), "ci_hi": r3(hi),
                              "p": r3(mc["p"]), "n": len(ids),
                              "n_better": mc["n10"], "n_worse": mc["n01"],
                              "equivalent_to_no_signal": tost["equivalent"],
                              "tost_lo": r3(tost["lo"]),
                              "tost_hi": r3(tost["hi"])}
            M["prior_reliance_gap"][m][b] = per

    # ------------------------------------------- scale-trend statistics --
    # Models run nested tiers, so the trend is computed ONLY on the items
    # every model actually ran -- otherwise a size effect would be confounded
    # with a difference in item sets.
    M["scale_trend"] = {}
    for b in BACKENDS:
        have = [m for m in present if (m, b) in corr]
        if len(have) < 3:
            continue
        common = set(corr[(have[0], b)])
        for m in have[1:]:
            common &= set(corr[(m, b)])
        base_common = common & set(corr.get((have[0], "no_signal"), {}))
        for m in have[1:]:
            base_common &= set(corr.get((m, "no_signal"), {}))
        ids = sorted(common)
        if len(ids) < 30:
            continue
        xs = [SIZE[m] for m in have]
        ys = [sum(corr[(m, b)][i] for i in ids) / len(ids) for m in have]
        ent = {"sizes_b": xs, "acc": [r3(y) for y in ys],
               "n_common_items": len(ids),
               "models": [SHORT[m] for m in have],
               "acc_vs_size": {k: r3(v) for k, v in
                               st.spearman_trend(xs, ys).items()}}
        bids = sorted(base_common)
        if b != "no_signal" and len(bids) >= 30:
            gaps = [sum(corr[(m, b)][i] - corr[(m, "no_signal")][i]
                        for i in bids) / len(bids) for m in have]
            ent["prior_gap"] = [r3(g) for g in gaps]
            ent["n_common_items_gap"] = len(bids)
            ent["gap_vs_size"] = {k: r3(v) for k, v in
                                  st.spearman_trend(xs, gaps).items()}
        M["scale_trend"][b] = ent

    # --------------------------------------- routing vs reading (tools) --
    # tool_forced  : the right tool is always executed  -> reading ability
    # tool_augmented: the model must choose the tool    -> + routing ability
    # feature_prompt: ALL measurements handed over      -> pure reading
    M["tool_decomposition"] = {}
    for m in present:
        if (m, "tool_augmented") not in corr or (m, "tool_forced") not in corr:
            continue
        ta, tf = corr[(m, "tool_augmented")], corr[(m, "tool_forced")]
        ids = sorted(set(ta) & set(tf))
        a, z = [ta[i] for i in ids], [tf[i] for i in ids]
        d, lo, hi = st.boot_diff_ci(a, z)
        mc = st.mcnemar(a, z)
        ent = {"routing_cost": r3(-d), "ci_lo": r3(-hi), "ci_hi": r3(-lo),
               "p": r3(mc["p"]), "n": len(ids),
               "tool_augmented_acc": r3(sum(a) / len(a)),
               "tool_forced_acc": r3(sum(z) / len(z))}
        if (m, "feature_prompt") in corr:
            fp = corr[(m, "feature_prompt")]
            ids2 = sorted(set(fp) & set(tf))
            ent["feature_prompt_acc"] = r3(
                sum(fp[i] for i in ids2) / len(ids2))
        if (m, "tool_distractor") in corr:
            td = corr[(m, "tool_distractor")]
            ids_d = sorted(set(td) & set(ta))
            if ids_d:
                ad = [td[i] for i in ids_d]
                zd = [ta[i] for i in ids_d]
                dd, dlo, dhi = st.boot_diff_ci(ad, zd)
                ent["distractor"] = {
                    "acc": r3(sum(ad) / len(ad)),
                    "tool_augmented_acc_paired": r3(sum(zd) / len(zd)),
                    "cost": r3(-dd), "ci_lo": r3(-dhi), "ci_hi": r3(-dlo),
                    "p": r3(st.mcnemar(ad, zd)["p"]), "n": len(ids_d)}
        oc = corr[("dsp_oracle", "dsp_oracle")]
        ids3 = sorted(set(tf) & set(oc))
        ent["reading_cost_vs_oracle"] = r3(
            sum(oc[i] for i in ids3) / len(ids3)
            - sum(tf[i] for i in ids3) / len(ids3))
        ent["per_family"] = {}
        for f in FAMILIES:
            fid = [i for i in ids if items[i]["family"] == f]
            if not fid:
                continue
            ent["per_family"][f] = {
                "tool_augmented": r3(sum(ta[i] for i in fid) / len(fid)),
                "tool_forced": r3(sum(tf[i] for i in fid) / len(fid)),
                "routing_cost": r3(sum(tf[i] - ta[i] for i in fid)
                                   / len(fid))}
        M["tool_decomposition"][m] = ent

    # ---------------------------------------------- counterfactual base --
    base_pred = dict(pred)

    def cf_runs(model, backend):
        if model == "dsp_oracle":
            return [dict(r, model="dsp_oracle", backend="dsp_oracle")
                    for r in oracle if r["section"] == "cf"]
        return [r for r in cf if r["model"] == model
                and r["backend"] == backend]

    def gt_moved(item, var_gt):
        if var_gt in (None, "None"):
            return True
        if item["tolerance"]["type"] == "exact":
            return str(var_gt).upper() != str(item["ground_truth"]).upper()
        g0, g1 = num(item["ground_truth"]), num(var_gt)
        return abs(g1 - g0) > band(item["tolerance"], g0)

    conds = [(m, b) for m in present for b in SIG_BACKENDS]
    conds.append(("dsp_oracle", "dsp_oracle"))

    # ------------------------------------------------- hallucination -----
    M["hallucination"] = {}
    for m, b in conds:
        runs = [r for r in cf_runs(m, b)
                if r["variant"] in ("signal_swap", "scramble")]
        if not runs:
            continue
        ent = {"per_type": {}}
        pooled, pooled_undef = [], 0
        sticky_n = sticky_hit = 0
        for t in ("signal_swap", "scramble"):
            tr = [r for r in runs if r["variant"] == t]
            flags, undef, excl = [], 0, 0
            for r in tr:
                it = items[r["item_id"]]
                if not gt_moved(it, r["ground_truth"]):
                    excl += 1
                    continue
                bp = base_pred.get((m, b), {}).get(r["item_id"])
                u = unchanged(it, bp, r["pred"]) if (
                    bp is not None and r["pred"] is not None) else None
                if u is None:
                    undef += 1
                else:
                    flags.append(int(u))
                if t == "scramble" and r["ground_truth"] in (None, "None"):
                    sticky_n += 1
                    gp, vp = num(it["ground_truth"]), num(r["pred"])
                    if gp is not None and vp is not None and \
                            abs(vp - gp) <= band(it["tolerance"], gp):
                        sticky_hit += 1
            if tr:
                e = ci_entry(flags)
                e["n_unparseable_pair"] = undef
                e["n_excluded_gt_unmoved"] = excl
                e["rate"] = e.pop("acc")
                ent["per_type"][t] = e
            pooled += flags
            pooled_undef += undef
        e = ci_entry(pooled)
        e["rate"] = e.pop("acc")
        e["n_unparseable_pair"] = pooled_undef
        ent["pooled"] = e
        if not pooled:            # nothing scored -> omit rather than nan
            continue
        if sticky_n:
            ent["scramble_answers_original_gt_rate"] = r3(sticky_hit
                                                          / sticky_n)
            ent["scramble_answers_original_gt_n"] = sticky_n
        M["hallucination"][f"{m}|{b}"] = ent

    # ------------------------------------- directional cf consistency ----
    M["cf_consistency"] = {}
    for m, b in conds:
        runs = [r for r in cf_runs(m, b) if r.get("expected_direction")]
        if not runs:
            continue
        ent = {"per_type": {}}
        pooled = []
        for t in sorted({r["variant"] for r in runs}):
            tr = [r for r in runs if r["variant"] == t]
            flags, undef = [], 0
            for r in tr:
                bp = base_pred.get((m, b), {}).get(r["item_id"])
                vp, d = r["pred"], r["expected_direction"]
                if bp is None or vp is None:
                    undef += 1
                    continue
                if d == "flip":
                    ok = str(vp).upper() != str(bp).upper()
                else:
                    bn, vn = num(bp), num(vp)
                    if bn is None or vn is None:
                        undef += 1
                        continue
                    ok = vn < bn if d == "decrease" else vn > bn
                flags.append(int(ok))
            e = ci_entry(flags)
            e["consistency"] = e.pop("acc")
            e["n_unparseable_pair"] = undef
            ent["per_type"][t] = e
            pooled += flags
        e = ci_entry(pooled)
        e["consistency"] = e.pop("acc")
        ent["pooled"] = e
        if not pooled:            # nothing scored -> omit rather than nan
            continue
        M["cf_consistency"][f"{m}|{b}"] = ent

    # ---------------------------------------------------- contamination --
    M["contamination"] = {}
    orc = {r["item_id"]: r["correct"] for r in oracle
           if r["section"] == "contam"}
    if orc:
        arms = defaultdict(list)
        for iid, c in orc.items():
            arms[items[iid]["meta"]["contamination_probe"]].append(c)
        M["contamination"]["dsp_oracle"] = {
            a: ci_entry(v) for a, v in arms.items()}

    def probe_block(runs, label):
        can = [r for r in runs if items[r["item_id"]]["meta"][
            "contamination_probe"] == "canonical"]
        per = [r for r in runs if items[r["item_id"]]["meta"][
            "contamination_probe"] == "perturbed"]
        if not can or not per:
            return None
        by_pair = defaultdict(dict)
        for r in runs:
            it = items[r["item_id"]]
            by_pair[it["meta"]["pair_id"]][
                it["meta"]["contamination_probe"]] = (r["correct"],
                                                      r["pred"], it)
        pc, pp, same, tot = [], [], 0, 0
        recite_n = recite_hit = 0
        for arms2 in by_pair.values():
            if "canonical" not in arms2 or "perturbed" not in arms2:
                continue
            (cc, cpred, it), (pcorr, ppred, pit) = (arms2["canonical"],
                                                    arms2["perturbed"])
            pc.append(cc)
            pp.append(pcorr)
            u = unchanged(it, cpred, ppred) if (cpred is not None
                                                and ppred is not None) else None
            if u is not None:
                tot += 1
                same += int(u)
            # The perturbation is ANSWER PRESERVING by construction
            # (rotation preserves magnitude, resampling preserves
            # frequency), so the two arms of a pair share a ground truth.
            # We measure that rather than assume it: if it holds, a gap
            # cannot be ground-truth drift, and a grounded system must
            # score identically on both arms -- which is what makes any
            # gap interpretable as surface-form sensitivity (of which
            # memorisation of the released text is one cause).
            cgt, pgt = num(it["ground_truth"]), num(pit["ground_truth"])
            if cgt is not None and pgt is not None:
                recite_n += 1
                if abs(cgt - pgt) <= band(it["tolerance"], cgt):
                    recite_hit += 1
        d, lo, hi = st.boot_diff_ci(pc, pp)
        mc = st.mcnemar(pc, pp)
        ent = {"label": label,
               "canonical": ci_entry(pc), "perturbed": ci_entry(pp),
               "memorization_gap": r3(d), "gap_ci_lo": r3(lo),
               "gap_ci_hi": r3(hi), "gap_p": r3(mc["p"]),
               "n_pairs": len(pc),
               "paired_answer_unchanged_rate": r3(same / tot) if tot else None,
               "n_pairs_scored": tot,
               "gt_preserved_rate": (r3(recite_hit / recite_n)
                                     if recite_n else None),
               "n_gt_compared": recite_n}
        ent["per_family"] = {}
        for fam in ("peak_count", "dominant_freq"):
            fc = [r["correct"] for r in can if r["family"] == fam]
            fp2 = [r["correct"] for r in per if r["family"] == fam]
            if fc and fp2:
                ent["per_family"][fam] = {
                    "canonical": ci_entry(fc), "perturbed": ci_entry(fp2),
                    "gap": r3(sum(fc) / len(fc) - sum(fp2) / len(fp2))}
        return ent

    for m in present:
        runs = [r for r in contam if r["model"] == m]
        ws = [r for r in runs if r["backend"] == "with_signal"]
        ns = [r for r in runs if r["backend"] == "no_signal"]
        pr = [r for r in runs if r["backend"] == "contam_primed"]
        if not ws:
            continue
        ent = {"audited": probe_block(ws, "with_signal")}
        if pr:
            ent["positive_control"] = probe_block(pr, "contam_primed")
        if ns:
            ent["no_signal_acc_on_canonical_questions"] = r3(
                sum(r["correct"] for r in ns) / len(ns))
        M["contamination"][m] = ent

    # ------------------------------------------------------ tool usage ---
    M["tool_usage"] = {}
    for m in present:
        runs = [r for r in grid if r["model"] == m
                and r["backend"] == "tool_augmented"]
        if not runs:
            continue
        calls = defaultdict(int)
        right = defaultdict(lambda: [0, 0])
        n_with = 0
        for r in runs:
            tc = r.get("tool_calls") or []
            n_with += bool(tc)
            want = {"peak_count": "count_peaks",
                    "dominant_freq": "dominant_freq",
                    "intensity_order": "sma",
                    "transition_time": "transition_detect",
                    "bout_duration": "active_bout"}[r["family"]]
            right[r["family"]][1] += 1
            if any(c["tool"] == want for c in tc):
                right[r["family"]][0] += 1
            for c in tc:
                calls[c["tool"]] += 1
        M["tool_usage"][m] = {
            "frac_runs_with_tool_call": r3(n_with / len(runs)),
            "calls_per_run": r3(sum(calls.values()) / len(runs)),
            "calls_by_tool": dict(sorted(calls.items(),
                                         key=lambda kv: -kv[1])),
            "correct_tool_rate": {f: r3(v[0] / v[1])
                                  for f, v in right.items()},
            "correct_tool_rate_overall": r3(
                sum(v[0] for v in right.values())
                / sum(v[1] for v in right.values()))}

    # ----------------------------------------------- format robustness ---
    M["format_robustness"] = {}
    fmt_models = sorted({r["model"] for r in fmt})
    for m in fmt_models:
        ent = {}
        ids_ws = corr.get((m, "with_signal"), {})
        for b in FORMAT_BACKENDS:
            runs = [r for r in fmt if r["model"] == m and r["backend"] == b]
            if not runs:
                continue
            e = ci_entry([r["correct"] for r in runs])
            ids = [r["item_id"] for r in runs if r["item_id"] in ids_ws]
            if ids:
                a = [next(r["correct"] for r in runs if r["item_id"] == i)
                     for i in ids]
                z = [ids_ws[i] for i in ids]
                d, lo, hi = st.boot_diff_ci(a, z)
                mc = st.mcnemar(a, z)
                e.update({"vs_triaxial": r3(d), "vs_ci_lo": r3(lo),
                          "vs_ci_hi": r3(hi), "vs_p": r3(mc["p"]),
                          "n_paired": len(ids),
                          "equivalent": st.tost_equivalence(
                              a, z, margin=EQUIV_MARGIN)["equivalent"]})
            e["mean_prompt_tokens"] = r3(sum(r["prompt_tokens"]
                                             for r in runs) / len(runs))
            ent[b] = e
        # the same items under the default triaxial rendering
        fmt_ids = sorted({r["item_id"] for r in fmt if r["model"] == m})
        base_ids = [i for i in fmt_ids if i in ids_ws]
        if base_ids:
            ent["with_signal_triaxial"] = ci_entry([ids_ws[i]
                                                    for i in base_ids])
        M["format_robustness"][m] = ent

    # ---------------------------------------------- decoding variance ----
    M["decode_variance"] = {}
    for m in sorted({r["model"] for r in dec}):
        per_seed, by_item = {}, defaultdict(list)
        for s in evalplan.DECODE_SEEDS:
            runs = [r for r in dec if r["model"] == m
                    and r.get("decode_seed") == s]
            if runs:
                per_seed[s] = r3(sum(r["correct"] for r in runs) / len(runs))
        for r in dec:
            if r["model"] == m:
                by_item[r["item_id"]].append(r["pred"])
        agree = [len({str(p) for p in v}) == 1
                 for v in by_item.values() if len(v) > 1]
        greedy = corr.get((m, "with_signal"), {})
        ids = [i for i in by_item if i in greedy]
        accs = list(per_seed.values())
        M["decode_variance"][m] = {
            "per_seed_acc": per_seed,
            "acc_range": r3(max(accs) - min(accs)) if accs else None,
            "mean_acc": r3(sum(accs) / len(accs)) if accs else None,
            "greedy_acc_same_items": r3(
                sum(greedy[i] for i in ids) / len(ids)) if ids else None,
            "identical_across_seeds_rate": r3(
                sum(agree) / len(agree)) if agree else None,
            "n_items": len(by_item)}

    # ------------------------------------------- degeneracy summary ------
    # Modal answer share is an accuracy-free grounding signal: an answer that
    # is the same on every item cannot be a function of the item's signal.
    # Averaged over the four numeric families (intensity_order is binary, so
    # its modal share has a 0.5 floor by construction).
    NUMERIC = [f for f in FAMILIES if f != "intensity_order"]
    M["modal_share"] = {"per_condition": {}, "numeric_families": NUMERIC}
    for b in BACKENDS:
        vals = []
        for m in present:
            e = M["grid"].get(m, {}).get(b)
            if not e:
                continue
            v = [e["diversity"][f]["modal_share"] for f in NUMERIC
                 if f in e.get("diversity", {})
                 and e["diversity"][f]["modal_share"] is not None]
            if v:
                vals.append(sum(v) / len(v))
        if vals:
            M["modal_share"]["per_condition"][b] = {
                "mean": r3(sum(vals) / len(vals)),
                "min": r3(min(vals)), "max": r3(max(vals)),
                "n_models": len(vals)}
    raw_b = [b for b in ("with_signal", "with_signal_cot",
                         "with_signal_long", "no_signal")
             if b in M["modal_share"]["per_condition"]]
    dsp_b = [b for b in ("feature_prompt", "tool_forced", "tool_augmented")
             if b in M["modal_share"]["per_condition"]]
    for tag, group in (("raw_prompted", raw_b), ("dsp_mediated", dsp_b)):
        if group:
            v = [M["modal_share"]["per_condition"][b]["mean"] for b in group]
            M["modal_share"][tag] = r3(sum(v) / len(v))

    # cells where a near-constant answer still scores well -- the reason a
    # bare accuracy figure is not interpretable
    hits = []
    for m in present:
        for b in BACKENDS:
            e = M["grid"].get(m, {}).get(b)
            if not e:
                continue
            for f in FAMILIES:
                d = e.get("diversity", {}).get(f)
                a2 = e["per_family"].get(f, {}).get("acc")
                if d and a2 is not None and d["modal_share"] >= 0.95 \
                        and a2 >= 0.5:
                    hits.append({"model": SHORT[m], "condition": b,
                                 "family": f, "acc": a2,
                                 "modal_share": d["modal_share"]})
    M["constant_answer_high_accuracy"] = hits

    with open(os.path.join(ROOT, "results", "metrics.json"), "w") as f:
        json.dump(M, f, indent=2)
    print("wrote results/metrics.json")
    write_md(M, present)
    print("wrote results/RESULTS.md")


# ------------------------------------------------------------ markdown ---

def _a(e, dec=3):
    if not e or e.get("acc") is None:
        return "--"
    return f"{e['acc']:.{dec}f} [{e['ci_lo']:.{dec}f},{e['ci_hi']:.{dec}f}]"


def _r(e, dec=3):
    if not e or e.get("rate") is None and e.get("consistency") is None:
        return "--"
    v = e.get("rate", e.get("consistency"))
    return f"{v:.{dec}f} [{e['ci_lo']:.{dec}f},{e['ci_hi']:.{dec}f}]"


def _p(p):
    if p is None:
        return "--"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def write_md(M, present):
    L = ["# GroundCheck Stage-3 audit results", "",
         f"Generated {M['timestamp']} by scripts/aggregate.py from "
         f"results/raw/*.jsonl.",
         "",
         f"Raw runs: {M['n_raw_runs']} (total {M['n_raw_total']}).",
         f"Models: {', '.join(M['models_present'])}.",
         "All rates carry Wilson 95% CIs; all gaps carry paired item-level "
         "bootstrap CIs (10k resamples) and exact McNemar p-values, "
         "Holm-corrected within each family battery.", ""]

    L += ["## 1. Tolerance-band accuracy [95% CI] by family and condition",
          ""]
    hdr = "| model | condition | " + " | ".join(FAMILIES) + " | overall |"
    L += [hdr, "|" + "---|" * (len(FAMILIES) + 3)]
    d = M["dsp_oracle"]
    L.append("| — | dsp_oracle (ceiling) | "
             + " | ".join(_a(d["per_family"][f]) for f in FAMILIES)
             + f" | {_a(d['overall'])} |")
    bc = M["best_constant"]
    L.append("| — | *best constant answer (floor)* | "
             + " | ".join(f"{bc['per_family'][f]['acc']:.3f} "
                          f"(={bc['per_family'][f]['answer']})"
                          for f in FAMILIES)
             + f" | {bc['overall']:.3f} |")
    for m in present:
        for b in BACKENDS:
            e = M["grid"].get(m, {}).get(b)
            if not e:
                continue
            cells = " | ".join(_a(e["per_family"].get(f)) for f in FAMILIES)
            L.append(f"| {SHORT[m]} | {b} | {cells} | {_a(e['overall'])} |")

    L += ["", "### 1b. transition_time by provenance "
              "(real PAMAP2 consensus vs spliced UCI-HAR)", "",
          "| model | condition | real | spliced |", "|---|---|---|---|"]
    L.append("| — | dsp_oracle | "
             + _a(M["dsp_oracle"]["per_provenance"].get("real")) + " | "
             + _a(M["dsp_oracle"]["per_provenance"].get("spliced")) + " |")
    for m in present:
        for b in BACKENDS:
            e = M["grid"].get(m, {}).get(b)
            if not e or not e.get("per_provenance"):
                continue
            L.append(f"| {SHORT[m]} | {b} | "
                     f"{_a(e['per_provenance'].get('real'))} | "
                     f"{_a(e['per_provenance'].get('spliced'))} |")

    L += ["", "## 2. Prior-reliance gap with paired tests", "",
          f"gap = acc(condition) − acc(no_signal), same items. "
          f"McNemar exact p; TOST equivalence margin ±{M['equivalence_margin']}.",
          "", "| model | condition | overall gap [95% CI] | p | equivalent to "
          "no_signal? |", "|---|---|---|---|---|"]
    for m in present:
        for b in BACKENDS:
            g = M["prior_reliance_gap"].get(m, {}).get(b, {}).get("overall")
            if not g:
                continue
            L.append(f"| {SHORT[m]} | {b} | {g['gap']:+.3f} "
                     f"[{g['ci_lo']:+.3f},{g['ci_hi']:+.3f}] | "
                     f"{_p(g['p'])} | "
                     f"{'YES' if g['equivalent_to_no_signal'] else 'no'} |")

    L += ["", "### 2b. Per-family prior gap (Holm-corrected within model)",
          "", "| model | condition | " + " | ".join(FAMILIES) + " |",
          "|" + "---|" * (len(FAMILIES) + 2)]
    for m in present:
        for b in BACKENDS:
            per = M["prior_reliance_gap"].get(m, {}).get(b)
            if not per:
                continue
            cells = []
            for f in FAMILIES:
                c = per.get(f)
                if not c:
                    cells.append("--")
                else:
                    star = "*" if c.get("significant") else ""
                    cells.append(f"{c['gap']:+.3f}{star}")
            L.append(f"| {SHORT[m]} | {b} | " + " | ".join(cells) + " |")
    L += ["", "`*` = significant after Holm correction (α=0.05).", ""]

    L += ["## 3. Scale trend (Spearman over model size)", "",
          "| condition | sizes (B) | overall acc | ρ(acc,size) | p | "
          "ρ(prior-gap,size) | p |", "|---|---|---|---|---|---|---|"]
    for b, e in M["scale_trend"].items():
        av = e.get("acc_vs_size", {})
        gv = e.get("gap_vs_size", {})
        L.append(f"| {b} | {e['n_common_items']} | {e['sizes_b']} | "
                 f"{e['acc']} | "
                 f"{av.get('rho')} | {_p(av.get('p'))} | "
                 f"{gv.get('rho', '--')} | {_p(gv.get('p'))} |")

    L += ["", "## 4. Hallucination under signal_swap / scramble", "",
          "Scored only where the true answer moved outside the base "
          "tolerance band and both answers parsed. Lower is better; the "
          "dsp_oracle row is the metric's noise floor.", "",
          "| model | condition | swap | scramble | pooled | n | "
          "scrambled→original GT |", "|---|---|---|---|---|---|---|"]
    for key, e in M["hallucination"].items():
        m, b = key.split("|")
        name = SHORT.get(m, m)
        L.append(f"| {name} | {b} | {_r(e['per_type'].get('signal_swap'))} | "
                 f"{_r(e['per_type'].get('scramble'))} | "
                 f"{_r(e['pooled'])} | {e['pooled']['n']} | "
                 f"{e.get('scramble_answers_original_gt_rate', '--')} |")

    L += ["", "## 5. Directional counterfactual consistency", "",
          "| model | condition | pooled [95% CI] | n | per-type |",
          "|---|---|---|---|---|"]
    for key, e in M["cf_consistency"].items():
        m, b = key.split("|")
        pt = ", ".join(f"{t}={v.get('consistency')}"
                       for t, v in sorted(e["per_type"].items()))
        L.append(f"| {SHORT.get(m, m)} | {b} | {_r(e['pooled'])} | "
                 f"{e['pooled']['n']} | {pt} |")

    L += ["", "## 6. Routing vs reading decomposition", "",
          "tool_forced executes the correct tool for the model (reading "
          "only); tool_augmented makes the model choose (routing + "
          "reading); feature_prompt hands over every measurement.", "",
          "| model | feature_prompt | tool_forced | tool_augmented | "
          "routing cost [95% CI] | p | reading cost vs oracle |",
          "|---|---|---|---|---|---|---|"]
    for m, e in M["tool_decomposition"].items():
        L.append(f"| {SHORT[m]} | {e.get('feature_prompt_acc', '--')} | "
                 f"{e['tool_forced_acc']} | {e['tool_augmented_acc']} | "
                 f"{e['routing_cost']:+.3f} [{e['ci_lo']:+.3f},"
                 f"{e['ci_hi']:+.3f}] | {_p(e['p'])} | "
                 f"{e['reading_cost_vs_oracle']:+.3f} |")

    L += ["", "## 7. Contamination probe (+ primed positive control)", "",
          "| model | arm-set | canonical | perturbed | gap [95% CI] | p | "
          "unchanged | GT preserved |",
          "|---|---|---|---|---|---|---|---|"]
    dso = M["contamination"].get("dsp_oracle")
    if dso:
        L.append(f"| dsp_oracle | — | {_a(dso.get('canonical'))} | "
                 f"{_a(dso.get('perturbed'))} | — | — | — |")
    for m in present:
        e = M["contamination"].get(m)
        if not e:
            continue
        for key in ("audited", "positive_control"):
            blk = e.get(key)
            if not blk:
                continue
            L.append(f"| {SHORT[m]} | {blk['label']} | "
                     f"{_a(blk['canonical'])} | {_a(blk['perturbed'])} | "
                     f"{blk['memorization_gap']:+.3f} "
                     f"[{blk['gap_ci_lo']:+.3f},{blk['gap_ci_hi']:+.3f}] | "
                     f"{_p(blk['gap_p'])} | "
                     f"{blk['paired_answer_unchanged_rate']} | "
                     f"{blk['gt_preserved_rate']} |")

    L += ["", "## 8. Tool usage", "",
          "| model | runs w/ ≥1 call | calls/run | correct-tool rate | "
          "calls by tool |", "|---|---|---|---|---|"]
    for m, e in M["tool_usage"].items():
        L.append(f"| {SHORT[m]} | {e['frac_runs_with_tool_call']} | "
                 f"{e['calls_per_run']} | {e['correct_tool_rate_overall']} | "
                 f"{json.dumps(e['calls_by_tool'])} |")

    L += ["", "## 9. Format robustness (same items, same information)", "",
          "| model | rendering | acc [95% CI] | Δ vs triaxial [95% CI] | p | "
          "equivalent? | prompt tok |", "|---|---|---|---|---|---|---|"]
    for m, e in M["format_robustness"].items():
        if "with_signal_triaxial" in e:
            L.append(f"| {SHORT.get(m, m)} | triaxial (default) | "
                     f"{_a(e['with_signal_triaxial'])} | — | — | — | — |")
        for b in FORMAT_BACKENDS:
            x = e.get(b)
            if not x:
                continue
            L.append(f"| {SHORT.get(m, m)} | {b.replace('with_signal_', '')} "
                     f"| {_a(x)} | "
                     f"{x.get('vs_triaxial', 0):+.3f} "
                     f"[{x.get('vs_ci_lo', 0):+.3f},"
                     f"{x.get('vs_ci_hi', 0):+.3f}] | "
                     f"{_p(x.get('vs_p'))} | "
                     f"{'YES' if x.get('equivalent') else 'no'} | "
                     f"{x['mean_prompt_tokens']:.0f} |")
    ofm = M["dsp_oracle"].get("per_format", {})
    if ofm:
        L += ["", "dsp_oracle per rendering (information check): "
              + ", ".join(f"{k}={v['acc']}" for k, v in ofm.items()), ""]

    L += ["", "## 10. Decoding variance (T=0.7, 3 seeds)", "",
          "| model | per-seed acc | range | greedy acc (same items) | "
          "identical across seeds |", "|---|---|---|---|---|"]
    for m, e in M["decode_variance"].items():
        L.append(f"| {SHORT.get(m, m)} | {e['per_seed_acc']} | "
                 f"{e['acc_range']} | {e['greedy_acc_same_items']} | "
                 f"{e['identical_across_seeds_rate']} |")

    L += ["", "## 11. Cost per query", "",
          "| model | condition | latency s | prompt tok | gen tok | "
          "parse | cap-hit |", "|---|---|---|---|---|---|---|"]
    for m in present:
        for b in BACKENDS:
            e = M["grid"].get(m, {}).get(b)
            if not e:
                continue
            L.append(f"| {SHORT[m]} | {b} | {e['mean_latency_s']:.3f} | "
                     f"{e['mean_prompt_tokens']:.0f} | "
                     f"{e['mean_gen_tokens']:.1f} | {e['parse_rate']:.3f} | "
                     f"{e['gen_cap_hit_rate']:.3f} |")

    L += ["", "## 12. Response diversity (degeneracy check)", "",
          "modal_share = fraction of items receiving the single most common "
          "answer; 1.0 means a constant output.", "",
          "| model | condition | " + " | ".join(FAMILIES) + " |",
          "|" + "---|" * (len(FAMILIES) + 2)]
    for m in present:
        for b in BACKENDS:
            e = M["grid"].get(m, {}).get(b)
            if not e or not e.get("diversity"):
                continue
            cells = " | ".join(
                f"{e['diversity'][f]['modal_share']:.2f}"
                if f in e["diversity"] else "--" for f in FAMILIES)
            L.append(f"| {SHORT[m]} | {b} | {cells} |")

    L += ["", "## 13. Benchmark calibration: floors, concentration, "
          "tolerance, split stability", "",
          "The oracle constant is fitted on the evaluated items (a "
          "concentration diagnostic); the deployable constant is fitted on "
          "the held-out remainder of the pool and scored out of sample (the "
          "baseline).", "",
          "| family | oracle constant | acc | deployable constant | acc | "
          "distinguishable bins | entropy (nats) |",
          "|" + "---|" * 7]
    for f in FAMILIES:
        bc = M["best_constant"]["per_family"][f]
        dp = M["best_constant"]["deployable_per_family"].get(f) or {}
        cc = M.get("concentration", {}).get(f, {})
        L.append(f"| {f} | {bc['answer']} | {bc['acc']:.3f} | "
                 f"{dp.get('answer', '--')} | "
                 f"{dp.get('acc', float('nan')):.3f} | "
                 f"{cc.get('distinguishable_bins', '--')} | "
                 f"{cc.get('entropy_nats', '--')} |")
    tw = M.get("tolerance_sweep", {})
    if tw:
        L += ["", "Tolerance sweep (all bands scaled):", "",
              "| bands | floor | ceiling | headroom |", "|" + "---|" * 4]
        for k in ("x0.5", "x1", "x2"):
            if k in tw:
                e = tw[k]
                L.append(f"| {k} | {e['floor']:.3f} | {e['ceiling']:.3f} | "
                         f"{e['headroom']:.3f} |")
    sr = M.get("split_robustness", {})
    if sr:
        L += ["", "Prior-reliance gap over disjoint half-splits:", "",
              "| model | gap (full) | min | max | sd | n halves |",
              "|" + "---|" * 6]
        for m, e in sr.items():
            L.append(f"| {m} | {e['gap_full']:+.3f} | "
                     f"{e['half_split_min']:+.3f} | "
                     f"{e['half_split_max']:+.3f} | {e['half_split_sd']:.3f} "
                     f"| {e['n_halves']} |")

    fb = M.get("full_benchmark", {})
    if fb:
        L += ["", "## 14. Full-pool runs (is the shared subset "
              "representative?)", "",
              "| model | condition | full acc | n | subset acc | n | delta |",
              "|" + "---|" * 7]
        for m, v in fb.items():
            for b, e in v.items():
                L.append(f"| {m} | {b} | {e['full_acc']:.3f} | "
                         f"{e['n_full']} | {e['subset_acc']:.3f} | "
                         f"{e['n_subset']} | {e['delta']:+.3f} |")

    dis = [(SHORT.get(m, m), v["distractor"])
           for m, v in M.get("tool_decomposition", {}).items()
           if v.get("distractor")]
    if dis:
        L += ["", "## 15. Routing with distractor tools", "",
              "Four plausible-but-inapplicable tools are interleaved with "
              "the five working ones, so the correct call cannot be found "
              "by matching the question's noun.", "",
              "| model | tool_augmented | tool_distractor | cost | 95% CI | "
              "p | n |", "|" + "---|" * 7]
        for m, e in dis:
            L.append(f"| {m} | {e['tool_augmented_acc_paired']:.3f} | "
                     f"{e['acc']:.3f} | {e['cost']:+.3f} | "
                     f"[{e['ci_lo']:+.3f},{e['ci_hi']:+.3f}] | "
                     f"{e['p']:.3g} | {e['n']} |")

    with open(os.path.join(ROOT, "results", "RESULTS.md"), "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
