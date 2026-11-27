#!/usr/bin/env python3
"""Analysis for the November 2026 revision (reviews #63A/B/C).

Reads the submitted raw records (results/raw, unchanged), the new-model
records (results/raw_rev2611) and the ground-truth validation rows
(results/rev2611/gt_validation_rows.jsonl). Writes
results/rev2611/analysis.json and paper/tables/macros_rev2611.tex plus the
revision tables. Safe to re-run while runs are in progress (reports n).
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import evalplan  # noqa: E402
import harness  # noqa: E402
import stats as st  # noqa: E402

FAM = evalplan.FAMILIES
NUMERIC = [f for f in FAM if f != "intensity_order"]
OLD = {"Qwen/Qwen2.5-0.5B-Instruct": ("Qwen2.5-0.5B", "2024", 0.5),
       "Qwen/Qwen2.5-1.5B-Instruct": ("Qwen2.5-1.5B", "2024", 1.5),
       "Qwen/Qwen2.5-3B-Instruct": ("Qwen2.5-3B", "2024", 3),
       "Qwen/Qwen2.5-7B-Instruct": ("Qwen2.5-7B", "2024", 7),
       "meta-llama/Llama-3.1-8B-Instruct": ("Llama-3.1-8B", "2024", 8),
       "google/gemma-2-9b-it": ("Gemma-2-9B", "2024", 9)}
NEW = {"qwen3-8b-thinkFalse": ("Qwen3-8B", "2025", 8.2),
       "gemma4-26b-a4b-thinkFalse": ("Gemma-4-26B-A4B", "2026", 25.2),
       "mistral-small3.1-24b": ("Mistral-Small-3.1-24B", "2025", 24),
       "gemma4-31b-thinkFalse": ("Gemma-4-31B", "2026", 30.7),
       "gpt-oss-120b-thinklow": ("gpt-oss-120b (low)", "2025", 116.8),
       "gpt-oss-120b-thinkhigh": ("gpt-oss-120b (high)", "2025", 116.8)}


def load(paths, key_model):
    out, seen = [], set()
    for p in paths:
        for line in open(p):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            k = (r.get(key_model), r["backend"], r["item_id"], r["variant"],
                 r.get("decode_seed"))
            if k in seen:
                continue
            seen.add(k)
            r["degenerate"] = harness.is_degenerate(r.get("raw_response"))
            r["_m"] = r.get(key_model)
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
    r = num(ref)
    return max(tol["value"] * abs(r if r is not None else 0),
               tol.get("min_abs", 0.0))


def r3(x):
    return None if x is None else round(float(x), 3)


def best_const(ev, fit):
    """Deployable constant: fitted on `fit` items, scored on `ev` items.

    Uses each item's own tolerance. (The submitted aggregate applied the
    first evaluated item's tolerance to the whole family, which scored the
    spliced transitions at +-2 s instead of their +-1 s; see the revision
    notes.)
    """
    if not ev:
        return None
    if ev[0]["tolerance"]["type"] == "exact":
        c = [str(i["ground_truth"]).upper() for i in fit]
        best = max(set(c), key=c.count)
        return sum(str(i["ground_truth"]).upper() == best for i in ev)

    def acc(c, pool):
        return sum(abs(c - num(i["ground_truth"])) <= band(
            i["tolerance"], i["ground_truth"]) for i in pool)
    cands = sorted({num(i["ground_truth"]) for i in fit})
    best = max(cands, key=lambda c: acc(c, fit))
    return acc(best, ev)


def const_items(ev, fit):
    """Per-item correctness of the deployable constant (as best_const)."""
    if not ev:
        return {}
    if ev[0]["tolerance"]["type"] == "exact":
        c = [str(i["ground_truth"]).upper() for i in fit]
        best = max(set(c), key=c.count)
        return {i["id"]: int(str(i["ground_truth"]).upper() == best)
                for i in ev}

    def ok(c, i):
        return abs(c - num(i["ground_truth"])) <= band(i["tolerance"],
                                                       i["ground_truth"])
    cands = sorted({num(i["ground_truth"]) for i in fit})
    best = max(cands, key=lambda c: sum(ok(c, i) for i in fit))
    return {i["id"]: int(ok(best, i)) for i in ev}


def main():
    items_l = evalplan.load_items(os.path.join(ROOT, "data", "benchmark",
                                               "items.jsonl"))
    items = {i["id"]: i for i in items_l}
    plan = evalplan.build_plan(items_l)
    main_ids = [i["id"] for i in plan["main"]]
    ids_C = {i["id"] for i in plan["main_C"]}
    ids_B = {i["id"] for i in plan["main_B"]}
    heldout = [i for i in plan["main_full"] if i["id"] not in set(main_ids)]

    old = load(sorted(glob.glob(os.path.join(ROOT, "results", "raw",
                                             "grid_*.jsonl"))), "model")
    old = [r for r in old if r["item_id"] in set(main_ids)]
    oracle = {}
    for line in open(os.path.join(ROOT, "results", "raw", "dsp_oracle.jsonl")):
        r = json.loads(line)
        if r.get("section") == "grid" and r["variant"] == "base":
            oracle[r["item_id"]] = r          # as scripts/aggregate.py
    new = load(sorted(glob.glob(os.path.join(ROOT, "results", "raw_rev2611",
                                             "grid_*.jsonl"))), "model_tag")
    new = [r for r in new if r["item_id"] in set(main_ids)]
    # New models are evaluated on the 150 shared items only; records on other
    # items (from runs before the switch to the shared subset) are ignored.
    new_full = [r for r in new if r["_m"] == "gpt-oss-120b-thinklow"]
    new = [r for r in new if r["_m"] not in NEW or r["item_id"] in ids_C]
    rows = old + new
    names = {**{k: v[0] for k, v in OLD.items()},
             **{k: v[0] for k, v in NEW.items()}}

    # ---- robust subset from the ground-truth validation -----------------
    val = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(
        ROOT, "results", "rev2611", "gt_validation_rows.jsonl"))}

    def robust(iid, strict=True):
        v = val.get(iid)
        if v is None:
            return False
        f = v["family"]
        if f == "peak_count":
            ks = [k for k in v["alts"]]
            n = sum(v["agree"][k] for k in ks)
            return n == len(ks) if strict else n > len(ks) / 2
        if f == "dominant_freq":
            return all(v["agree"][k] for k in ("periodogram", "welch_2s",
                                                 "welch_8s"))
        if f == "intensity_order":
            return all(v["agree"].values())
        if f == "transition_time":
            if "detected" in v:                       # spliced
                return bool(v["agree"]["energy_changepoint"])
            return bool(v["agree"]["shown_placement_changepoint"])
        return True       # bout_duration: exact by construction
    rob = {i for i in main_ids if robust(i)}
    rob_major = {i for i in main_ids if robust(i, strict=False)}

    def acc_on(m, b, idset):
        rs = [r for r in rows if r["_m"] == m and r["backend"] == b
              and r["item_id"] in idset]
        if not rs:
            return None, 0
        return sum(r["correct"] for r in rs) / len(rs), len(rs)

    def floor_on(idset):
        tot = n = 0
        for f in FAM:
            ev = [items[i] for i in main_ids
                  if i in idset and items[i]["family"] == f]
            fit = [i for i in heldout if i["family"] == f]
            if ev:
                tot += best_const(ev, fit)
                n += len(ev)
        return tot / n if n else None

    def oracle_on(idset):
        v = [oracle[i]["correct"] for i in idset if i in oracle]
        return sum(v) / len(v) if v else None

    def modal(m, b, idset):
        out = []
        for f in NUMERIC:
            preds = [r["pred"] for r in rows if r["_m"] == m
                     and r["backend"] == b and r["family"] == f
                     and r["item_id"] in idset]
            if preds:
                out.append(st.response_diversity(preds)["modal_share"])
        return float(np.mean(out)) if out else None

    def gap(m, idset, b="with_signal"):
        a = {r["item_id"]: r["correct"] for r in rows if r["_m"] == m
             and r["backend"] == b and r["item_id"] in idset}
        z = {r["item_id"]: r["correct"] for r in rows if r["_m"] == m
             and r["backend"] == "no_signal" and r["item_id"] in idset}
        ids = sorted(set(a) & set(z))
        if len(ids) < 10:
            return None
        x = [a[i] for i in ids]
        y = [z[i] for i in ids]
        d, lo, hi = st.boot_diff_ci(x, y, paired=True)
        p = st.mcnemar(x, y)
        p = p.get("p") if isinstance(p, dict) else p
        return {"gap": r3(d), "lo": r3(lo), "hi": r3(hi), "p": p,
                "n": len(ids)}

    models_present = [m for m in list(OLD) + list(NEW)
                      if any(r["_m"] == m for r in rows)]
    conds = ["no_signal", "with_signal", "feature_prompt", "tool_forced",
             "tool_augmented", "code_agent"]
    A = {"robust_counts": {"strict": len(rob), "majority": len(rob_major),
                           "strict_C": len(rob & ids_C),
                           "per_family_strict": {f: sum(items[i]["family"]
                                                        == f for i in rob)
                                                 for f in FAM}},
         "floor": {"main": r3(floor_on(set(main_ids))),
                   "C": r3(floor_on(ids_C)), "B": r3(floor_on(ids_B)),
                   "robust": r3(floor_on(rob)),
                   "robust_C": r3(floor_on(rob & ids_C))},
         "oracle": {"main": r3(oracle_on(set(main_ids))),
                    "C": r3(oracle_on(ids_C)),
                    "robust": r3(oracle_on(rob)),
                    "robust_C": r3(oracle_on(rob & ids_C))},
         "models": {}}
    for m in models_present:
        ent = {"name": names[m], "conds": {}}
        for b in conds:
            a_all, n_all = acc_on(m, b, set(main_ids))
            if not n_all:
                continue
            a_c, n_c = acc_on(m, b, ids_C)
            a_r, n_r = acc_on(m, b, rob & ids_C)   # shared items only
            ent["conds"][b] = {"acc": r3(a_all), "n": n_all,
                               "acc_C": r3(a_c), "n_C": n_c,
                               "acc_robust": r3(a_r), "n_robust": n_r,
                               "modal_C": r3(modal(m, b, ids_C)),
                               "per_family_C": {
                                   f: r3(acc_on(m, b, {i for i in ids_C
                                                       if items[i]["family"]
                                                       == f})[0])
                                   for f in FAM}}
        ent["gap_C"] = gap(m, ids_C)
        ent["gap_all"] = gap(m, set(main_ids))
        sel = {r["item_id"] for r in rows if r["_m"] == m
               and r["backend"] == "with_signal"}
        ent["floor_own"] = r3(floor_on(sel)) if sel else None
        ent["oracle_own"] = r3(oracle_on(sel)) if sel else None
        A["models"][m] = ent

    # ---- Holm over models, and paired gains per family -------------------
    complete = [m for m in models_present
                if A["models"][m]["conds"].get("with_signal", {}).get("n_C")
                == len(ids_C) and A["models"][m]["conds"].get(
                    "no_signal", {}).get("n_C") == len(ids_C)]
    A["complete_models"] = complete
    ps = [A["models"][m]["gap_all"]["p"] for m in complete]
    adj, rej = st.holm(ps)
    for m, a_, r_ in zip(complete, adj, rej):
        A["models"][m]["gap_all"]["holm_p"] = float(a_)
        A["models"][m]["gap_all"]["holm_sig"] = bool(r_)
    cells = []
    for m in complete:
        for f in FAM:
            g = gap(m, {i for i in ids_C if items[i]["family"] == f})
            if g:
                cells.append((m, f, g))
    adj, rej = st.holm([g["p"] for _, _, g in cells])
    FG = {}
    for (m, f, g), a_, r_ in zip(cells, adj, rej):
        g["holm_p"], g["holm_sig"] = float(a_), bool(r_)
        FG.setdefault(m, {})[f] = g
    A["family_gaps"] = FG
    # modal share per family (no-signal vs raw text): a constant guess has
    # share 1; item-specific answers have low share (correctness-free)
    MF = {}
    for m in complete:
        for b in ("no_signal", "with_signal"):
            for f in FAM:
                preds = [r["pred"] for r in rows if r["_m"] == m
                         and r["backend"] == b and r["family"] == f
                         and r["item_id"] in ids_C]
                if preds:
                    MF.setdefault(m, {}).setdefault(b, {})[f] = r3(
                        st.response_diversity(preds)["modal_share"])
    A["modal_family_C"] = MF
    # generations that hit the token cap (new models record done_reasons)
    CAP = {}
    for m in models_present:
        for b in conds:
            rs = [r for r in rows if r["_m"] == m and r["backend"] == b
                  and r["item_id"] in ids_C]
            if not rs or b == "tool_augmented" or (b == "code_agent"
                                                   and m in OLD):
                continue
            if m in OLD:     # HF runs: 128-token answers, no done_reason
                hit = lambda r: (r.get("gen_tokens") or 0) >= 128
            elif all(r.get("llm_meta") for r in rs):
                hit = lambda r: any(d == "length" for d in
                                    r["llm_meta"].get("done_reasons", []))
            else:
                continue
            CAP.setdefault(m, {})[b] = {
                "all": r3(np.mean([hit(r) for r in rs])),
                **{f: int(sum(hit(r) for r in rs if r["family"] == f))
                   for f in FAM}}
    A["cap_C"] = CAP
    # own code vs raw text, paired, per model (Holm over models)
    cvr = []
    for m in complete:
        a_ = {r["item_id"]: r["correct"] for r in rows if r["_m"] == m
              and r["backend"] == "code_agent" and r["item_id"] in ids_C}
        z_ = {r["item_id"]: r["correct"] for r in rows if r["_m"] == m
              and r["backend"] == "with_signal" and r["item_id"] in ids_C}
        k = sorted(set(a_) & set(z_))
        if len(k) < 100:
            continue
        d, lo, hi = st.boot_diff_ci([a_[i] for i in k], [z_[i] for i in k],
                                    paired=True)
        cvr.append((m, {"gap": r3(d), "lo": r3(lo), "hi": r3(hi), "p":
                        st.mcnemar([a_[i] for i in k],
                                   [z_[i] for i in k])["p"], "n": len(k)}))
    if cvr:
        adj, rej = st.holm([v["p"] for _, v in cvr])
        for (m, v), a2, r2 in zip(cvr, adj, rej):
            v["holm_p"], v["holm_sig"] = float(a2), bool(r2)
    A["code_vs_raw"] = dict(cvr)

    # ---- paired tests against the constant, and gpt-oss on all items ----
    CONST = {}
    for f in FAM:
        ev = [items[i] for i in main_ids if items[i]["family"] == f]
        fit = [i for i in heldout if i["family"] == f]
        CONST.update(const_items(ev, fit))

    def vs_const(corr, ids):
        k = sorted(i for i in ids if i in corr and i in CONST)
        if len(k) < 10:
            return None
        x = [corr[i] for i in k]
        y = [CONST[i] for i in k]
        d, lo, hi = st.boot_diff_ci(x, y, paired=True)
        return {"gap": r3(d), "lo": r3(lo), "hi": r3(hi),
                "p": st.mcnemar(x, y)["p"], "n": len(k)}
    for m in complete:
        for b in ("no_signal", "with_signal"):
            corr = {r["item_id"]: r["correct"] for r in rows if r["_m"] == m
                    and r["backend"] == b}
            A["models"][m]["conds"][b]["vs_const_C"] = vs_const(corr, ids_C)
    # Holm over the vs-constant tests of the complete models, per condition
    for b in ("with_signal", "no_signal"):
        vc = [(m, A["models"][m]["conds"][b]["vs_const_C"]) for m in complete]
        adj, rej = st.holm([v["p"] for _, v in vc])
        for (m, v), a_, r_ in zip(vc, adj, rej):
            v["holm_p"], v["holm_sig"] = float(a_), bool(r_)
    G = "gpt-oss-120b-thinklow"
    ns = {r["item_id"]: r["correct"] for r in new_full
          if r["backend"] == "no_signal"}
    ws = {r["item_id"]: r["correct"] for r in new_full
          if r["backend"] == "with_signal"}
    both = set(ns) & set(ws)
    if len(both) > 150:
        k = sorted(both)
        d, lo, hi = st.boot_diff_ci([ws[i] for i in k], [ns[i] for i in k],
                                    paired=True)
        A["gptoss_full"] = {
            "n": len(k), "no_signal": r3(np.mean([ns[i] for i in k])),
            "with_signal": r3(np.mean([ws[i] for i in k])),
            "gap": r3(d), "lo": r3(lo), "hi": r3(hi),
            "p": st.mcnemar([ws[i] for i in k], [ns[i] for i in k])["p"],
            "floor": r3(np.mean([CONST[i] for i in k])),
            "raw_vs_const": vs_const(ws, both),
            "nosig_vs_const": vs_const(ns, both)}
        k2 = sorted(both - ids_C)
        d2, lo2, hi2 = st.boot_diff_ci([ws[i] for i in k2],
                                       [ns[i] for i in k2], paired=True)
        A["gptoss_full"]["outside_shared"] = {
            "n": len(k2), "gap": r3(d2), "lo": r3(lo2), "hi": r3(hi2),
            "p": st.mcnemar([ws[i] for i in k2], [ns[i] for i in k2])["p"]}

    # ---- bridge: Llama-3.1-8B through the new models' pipeline -----------
    BR = "llama3.1-8b-ollama-bridge"
    HFL = "meta-llama/Llama-3.1-8B-Instruct"
    br = [r for r in new if r["_m"] == BR and r["item_id"] in ids_C]
    if br:
        out = {}
        for b in ("no_signal", "with_signal", "feature_prompt"):
            rb = [r for r in br if r["backend"] == b]
            rh = [r for r in rows if r["_m"] == HFL and r["backend"] == b
                  and r["item_id"] in ids_C]
            if not rb:
                continue
            capb = np.mean([any(d == "length" for d in
                                (r.get("llm_meta") or {}).get(
                                    "done_reasons", [])) for r in rb])
            capb_h = np.mean([(r.get("gen_tokens") or 0) >= 128 for r in rh])
            out[b] = {"n": len(rb), "acc_ollama": r3(np.mean(
                [r["correct"] for r in rb])), "acc_hf": r3(np.mean(
                    [r["correct"] for r in rh])) if rh else None,
                "cap_ollama": r3(capb), "cap_hf": r3(capb_h)}
        nb = {r["item_id"]: r["correct"] for r in br
              if r["backend"] == "no_signal"}
        wb = {r["item_id"]: r["correct"] for r in br
              if r["backend"] == "with_signal"}
        k = sorted(set(nb) & set(wb))
        if len(k) >= 100:
            d, lo, hi = st.boot_diff_ci([wb[i] for i in k], [nb[i] for i in k],
                                        paired=True)
            out["gap"] = {"gap": r3(d), "lo": r3(lo), "hi": r3(hi),
                          "p": st.mcnemar([wb[i] for i in k],
                                          [nb[i] for i in k])["p"],
                          "n": len(k)}
        fam = {}
        for f in FAM:
            n0 = [r["correct"] for r in br if r["backend"] == "no_signal"
                  and r["family"] == f]
            w0 = [r["correct"] for r in br if r["backend"] == "with_signal"
                  and r["family"] == f]
            preds = [r["pred"] for r in br if r["backend"] == "with_signal"
                     and r["family"] == f]
            top = max(set(map(str, preds)), key=lambda v: [str(x) for x in
                                                         preds].count(v))
            fam[f] = {"nosig": r3(np.mean(n0)), "raw": r3(np.mean(w0)),
                      "modal_answer": top,
                      "modal_count": [str(x) for x in preds].count(top),
                      "n": len(preds)}
        out["per_family"] = fam
        A["bridge_llama"] = out

    # ---- families whose reference is not a DSP routine ------------------
    ind_ids = {i for i in ids_C if items[i]["family"] in
               ("transition_time", "bout_duration")}
    raw_v = [acc_on(m, "with_signal", ind_ids)[0] for m in models_present]
    giv_v = [acc_on(m, "tool_forced", ind_ids)[0] for m in models_present]
    raw_v = [v for v in raw_v if v is not None]
    giv_v = [v for v in giv_v if v is not None]
    A["independent_families"] = {
        "n": len(ind_ids), "raw_max": r3(max(raw_v)) if raw_v else None,
        "given_min": r3(min(giv_v)) if giv_v else None,
        "given_max": r3(max(giv_v)) if giv_v else None,
        "oracle": r3(oracle_on(ind_ids)), "floor": r3(floor_on(ind_ids))}
    A["oracle_per_family_C"] = {
        f: r3(oracle_on({i for i in ids_C if items[i]["family"] == f}))
        for f in FAM}

    # ---- high-effort probe: paired with low effort on the same items -----
    H, L = "gpt-oss-120b-thinkhigh", "gpt-oss-120b-thinklow"
    hid = {r["item_id"] for r in rows if r["_m"] == H
           and r["backend"] == "with_signal"} & ids_C
    if hid:
        hr = [r for r in rows if r["_m"] == H and r["backend"] == "with_signal"
              and r["item_id"] in hid]
        A["high_probe"] = {
            "n": len(hid),
            "mean_gen_tokens": r3(np.mean([r.get("gen_tokens") or 0
                                           for r in hr])),
            "n_capped": int(sum(any(d == "length" for d in
                                    (r.get("llm_meta") or {}).get(
                                        "done_reasons", [])) for r in hr)),
            "n_no_answer": int(sum(r["pred"] is None for r in hr)),
            "high_raw": r3(acc_on(H, "with_signal", hid)[0]),
            "low_raw": r3(acc_on(L, "with_signal", hid)[0]),
            "low_nosig": r3(acc_on(L, "no_signal", hid)[0]),
            "floor": r3(floor_on(hid)), "oracle": r3(oracle_on(hid))}

    # ---- code_agent detail ---------------------------------------------
    CA = {}
    for m in models_present:
        rs = [r for r in rows if r["_m"] == m and r["backend"] == "code_agent"]
        if not rs:
            continue
        ok_run = [any(c["status"] == "ok" for c in r.get("code_runs", []))
                  for r in rs]
        no_code = [len(r.get("code_runs", [])) == 0 for r in rs]
        per_f = {}
        for f in FAM:
            fr = [r for r in rs if r["family"] == f]
            if not fr:
                continue
            agree_or = []
            for r in fr:
                o = oracle.get(r["item_id"])
                if o is None or r["pred"] is None:
                    continue
                it = items[r["item_id"]]
                if it["tolerance"]["type"] == "exact":
                    agree_or.append(str(r["pred"]).upper() ==
                                    str(o["pred"]).upper())
                else:
                    on = num(o["pred"])
                    agree_or.append(on is not None and abs(num(r["pred"]) - on)
                                    <= band(it["tolerance"], on))
            per_f[f] = {"acc": r3(np.mean([r["correct"] for r in fr])),
                        "n": len(fr),
                        "agree_with_dsp_oracle": r3(np.mean(agree_or))
                        if agree_or else None,
                        "answered_without_code": r3(np.mean(
                            [len(r.get("code_runs", [])) == 0 for r in fr]))}
        CA[m] = {"n": len(rs), "acc": r3(np.mean([r["correct"] for r in rs])),
                 "any_successful_run": r3(np.mean(ok_run)),
                 "answered_without_code": r3(np.mean(no_code)),
                 "mean_runs": r3(np.mean([len(r.get("code_runs", []))
                                          for r in rs])),
                 "parse_rate": r3(np.mean([r["pred"] is not None
                                           for r in rs])),
                 "mean_latency_s": r3(np.mean([r["latency_s"] for r in rs])),
                 "per_family": per_f}
    A["code_agent"] = CA

    # ---- reasoning statistics for thinking models -----------------------
    TH = {}
    for m in models_present:
        rs = [r for r in rows if r["_m"] == m and "llm_meta" in r]
        if not rs:
            continue
        by_b = defaultdict(list)
        for r in rs:
            by_b[r["backend"]].append(r)
        TH[m] = {b: {"mean_thinking_tokens_est": r3(np.mean(
            [r["llm_meta"]["thinking_tokens_est"] for r in v])),
            "empty_content_rate": r3(np.mean(
                [r["llm_meta"]["empty_content"] > 0 for r in v])),
            "length_cap_rate": r3(np.mean(
                ["length" in (r["llm_meta"]["done_reasons"] or []) for r in v])),
            "mean_latency_s": r3(np.mean([r["latency_s"] for r in v]))}
            for b, v in by_b.items()}
    A["generation_stats"] = TH

    # ---- counterfactuals for new models ---------------------------------
    cf = load(sorted(glob.glob(os.path.join(ROOT, "results", "raw_rev2611",
                                            "cf_*.jsonl"))), "model_tag")
    CF = {}
    # gpt-oss's prompt contains the run date, and its counterfactual edits
    # ran on 26 Nov while its grid ran on 25 Nov. Raw-text edits are scored
    # against base answers rerun with the date pinned to 26 Nov; its tool and
    # own-code edits have date-mismatched bases and are flagged.
    GPT, GB = "gpt-oss-120b-thinklow", "gpt-oss-120b-thinklow-cfbase26"
    gbase = {r["item_id"]: r["pred"] for r in rows if r["_m"] == GB
             and r["backend"] == "with_signal"}
    for m in sorted({r["_m"] for r in cf}):
        for b in sorted({r["backend"] for r in cf if r["_m"] == m}):
            base = {r["item_id"]: r["pred"] for r in rows if r["_m"] == m
                    and r["backend"] == b}
            mismatch = False
            if m == GPT:
                if b == "with_signal" and gbase:
                    base = gbase
                else:
                    mismatch = True
            hall, cons = [], []
            cons_f = {}
            for r in cf:
                if r["_m"] != m or r["backend"] != b:
                    continue
                it = items[r["item_id"]]
                bp, vp = base.get(r["item_id"]), r["pred"]
                if bp is None or vp is None:
                    continue
                d = r.get("expected_direction")
                if d:
                    ok = None
                    if d == "flip":
                        ok = str(vp).upper() != str(bp).upper()
                    else:
                        bn, vn = num(bp), num(vp)
                        if bn is not None and vn is not None:
                            ok = vn < bn if d == "decrease" else vn > bn
                    if ok is not None:
                        cons.append(ok)
                        cons_f.setdefault(r["family"], []).append(ok)
                if r["variant"] in ("signal_swap", "scramble"):
                    vg = r["ground_truth"]
                    if vg in (None, "None"):
                        moved = True
                    elif it["tolerance"]["type"] == "exact":
                        moved = str(vg).upper() != str(
                            it["ground_truth"]).upper()
                    else:
                        g0 = num(it["ground_truth"])
                        moved = abs(num(vg) - g0) > band(it["tolerance"], g0)
                    if not moved:
                        continue
                    if it["tolerance"]["type"] == "exact":
                        hall.append(str(vp).upper() == str(bp).upper())
                    else:
                        bn, vn = num(bp), num(vp)
                        if bn is not None and vn is not None:
                            hall.append(abs(vn - bn) <= band(it["tolerance"],
                                                             bn))
            CF[f"{m}|{b}"] = {
                "halluc": r3(np.mean(hall)) if hall else None,
                "n_halluc": len(hall),
                "consistency": r3(np.mean(cons)) if cons else None,
                "n_cons": len(cons),
                "consistency_by_family": {f: [r3(np.mean(v)), len(v)]
                                          for f, v in cons_f.items()},
                "date_mismatch": mismatch}
    A["counterfactual"] = CF
    # DSP reference on the very same (item, variant) pairs the new models ran
    ref = {"event": [], "time_stretch": []}
    for it in plan["cf_B"]:
        b0 = num(harness.run_dsp_oracle(it)["pred"])
        for v in it["counterfactual_variants"]:
            if v["type"] in ("transition_shift", "bout_stretch",
                             "time_stretch"):
                v0 = num(harness.run_dsp_oracle(it, v)["pred"])
                if b0 is None or v0 is None:
                    continue
                ok = v0 < b0 if v["expected_direction"] == "decrease" \
                    else v0 > b0
                ref["time_stretch" if v["type"] == "time_stretch"
                    else "event"].append(ok)
    A["cf_reference_same_pairs"] = {k: [r3(np.mean(x)), len(x)]
                                    for k, x in ref.items() if x}
    # date sensitivity of gpt-oss: same items, same model, date 25 vs 26 Nov
    g25 = {r["item_id"]: r for r in rows if r["_m"] == GPT
           and r["backend"] == "with_signal"}
    g26 = {r["item_id"]: r for r in rows if r["_m"] == GB
           and r["backend"] == "with_signal"}
    k = sorted(set(g25) & set(g26))
    if k:
        A["date_sensitivity"] = {
            "n": len(k),
            "identical_pred": r3(np.mean([g25[i]["pred"] == g26[i]["pred"]
                                          for i in k])),
            "acc_25": r3(np.mean([g25[i]["correct"] for i in k])),
            "acc_26": r3(np.mean([g26[i]["correct"] for i in k]))}

    # ---- wording correction check --------------------------------------
    WTAG = {"q0.5b-wording": "Qwen/Qwen2.5-0.5B-Instruct",
            "q1.5b-wording": "Qwen/Qwen2.5-1.5B-Instruct",
            "q3b-wording": "Qwen/Qwen2.5-3B-Instruct",
            "q7b-wording": "Qwen/Qwen2.5-7B-Instruct",
            "llama8b-wording": "meta-llama/Llama-3.1-8B-Instruct",
            "gemma9b-wording": "google/gemma-2-9b-it",
            # gpt-oss rerun with the prompt date pinned to the grid's date
            "gpt-oss-120b-thinklow-d25-wording": "gpt-oss-120b-thinklow"}
    wrows = load(sorted(glob.glob(os.path.join(
        ROOT, "results", "raw_rev2611", "grid_*-wording.jsonl"))),
        "model_tag")
    WD = {}
    wtags = sorted({r["_m"] for r in wrows})
    if "gpt-oss-120b-thinklow-d25-wording" in wtags:   # date-matched rerun
        wtags.remove("gpt-oss-120b-thinklow-wording")
    for wt in wtags:
        m = WTAG.get(wt, wt[:-len("-wording")])
        for b in ("no_signal", "with_signal"):
            corr = {r["item_id"]: r["correct"] for r in wrows
                    if r["_m"] == wt and r["backend"] == b}
            orig = {r["item_id"]: r["correct"] for r in rows
                    if r["_m"] == m and r["backend"] == b}
            ids = sorted(set(corr) & set(orig))
            if not ids:
                continue
            x = [corr[i] for i in ids]
            y = [orig[i] for i in ids]
            d, lo, hi = st.boot_diff_ci(x, y, paired=True)
            WD[f"{m}|{b}"] = {"name": names.get(m, m), "n": len(ids),
                              "orig": r3(np.mean(y)), "corrected":
                              r3(np.mean(x)), "delta": r3(d), "lo": r3(lo),
                              "hi": r3(hi),
                              "p": st.mcnemar(x, y)["p"]}
    A["wording"] = WD

    out = os.path.join(ROOT, "results", "rev2611", "analysis.json")
    with open(out, "w") as f:
        json.dump(A, f, indent=1)
    print(json.dumps({"robust": A["robust_counts"], "floor": A["floor"],
                      "oracle": A["oracle"]}, indent=1))
    for m, e in A["models"].items():
        print(f"{e['name']:24s}", " ".join(
            f"{b[:8]}={c['acc_C']}/{c['n_C']}" for b, c in e["conds"].items()),
            "gapC", e["gap_C"] and (e["gap_C"]["gap"], e["gap_C"]["hi"]))
    for m, e in CA.items():
        print("CA", names[m], e["acc"], e["any_successful_run"],
              e["answered_without_code"],
              {f: (v["acc"], v["agree_with_dsp_oracle"])
               for f, v in e["per_family"].items()})
    print(json.dumps(CF, indent=0))


if __name__ == "__main__":
    main()
