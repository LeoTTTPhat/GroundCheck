"""Deterministic Stage-3 evaluation plan (shared by runner + aggregator).

Fixed subsets, seed 1.  Two model tiers so the audit can span 0.5B-9B on a
single workstation:

  tier A (<=7B, fast)  : the full 500-item main subset
  tier B (8-9B, slow)  : a nested 250-item half of the SAME subset, so tier-B
                         numbers are directly comparable to tier-A numbers on
                         those items (`main_ids_B` is a subset of `main_ids`).

Sections:
  main    : per-family cap over non-contamination items, every condition
  cf      : first CF_CAP main items per family -> all counterfactual variants,
            under with_signal + tool_augmented
  contam  : all contamination probes (with_signal) + canonical arm under
            no_signal + canonical/perturbed arms under the PRIMED positive
            control
  format  : format-robustness subset (magnitude / indexed / coarse / few-shot)
  decode  : decoding-variance subset (3 sampled decodes)
"""
import json

import numpy as np

FAMILIES = ["peak_count", "dominant_freq", "intensity_order",
            "transition_time", "bout_duration"]

MAIN_CAP = {"peak_count": 60, "dominant_freq": 60, "intensity_order": 60,
            "transition_time": 60, "bout_duration": 60}
# transition_time is split evenly between real (PAMAP2, leave-one-placement-
# out consensus ground truth) and spliced (UCI-HAR) provenance so the
# constructed-vs-real validity analysis has equal n on both arms.
TRANSITION_SPLIT = {"real": 30, "spliced": 30}

TIER_B_CAP = {"peak_count": 40, "dominant_freq": 40, "intensity_order": 40,
              "transition_time": 40, "bout_duration": 40}
# tier C: a further-nested 150-item subset for the slowest models, so their
# numbers stay comparable to A and B on common items
# (main_ids_C subset of main_ids_B subset of main_ids).
TIER_C_CAP = {"peak_count": 30, "dominant_freq": 30, "intensity_order": 30,
              "transition_time": 30, "bout_duration": 30}

CF_CAP = 30          # main items per family carried into the CF section
CF_CAP_B = 15        # ... for the slow tiers
FORMAT_CAP = 30      # main items per family in the format-robustness section
DECODE_CAP = 20      # main items per family in the decoding-variance section
DECODE_SEEDS = (0, 1, 2)
SEED = 1


def load_items(path):
    return [json.loads(l) for l in open(path)]


def full_pool(items):
    """Every non-contamination item (R4: split-robustness / full-benchmark)."""
    return [i for i in items if not i["meta"].get("contamination_probe")]


def build_plan(items, seed=SEED):
    rng = np.random.default_rng(seed)
    main = []
    for fam in FAMILIES:
        pool = [i for i in items if i["family"] == fam
                and not i["meta"].get("contamination_probe")]
        if fam == "transition_time":
            arms = {}
            for prov, cap in TRANSITION_SPLIT.items():
                sub = [i for i in pool
                       if bool(i["meta"].get("real_transition")) ==
                       (prov == "real")]
                perm = rng.permutation(len(sub))
                arms[prov] = [sub[j] for j in perm[:cap]]
            # interleave so that EVERY prefix of the transition block is
            # balanced between provenances -- tier B/C, cf, format and
            # decode are all prefixes, and a provenance-skewed prefix would
            # silently change what those sections measure.
            sel = []
            for k in range(max(len(v) for v in arms.values())):
                for prov in ("real", "spliced"):
                    if k < len(arms[prov]):
                        sel.append(arms[prov][k])
        else:
            perm = rng.permutation(len(pool))
            sel = [pool[j] for j in perm[:MAIN_CAP[fam]]]
        main.extend(sel)

    def head(cap_map, per_family=None):
        out = []
        for fam in FAMILIES:
            fam_items = [i for i in main if i["family"] == fam]
            k = per_family if per_family is not None else cap_map[fam]
            out.extend(fam_items[:k])
        return out

    contam = [i for i in items if i["meta"].get("contamination_probe")]
    canon = [i for i in contam
             if i["meta"]["contamination_probe"] == "canonical"]
    return {"main": main,
            "main_full": full_pool(items),
            "main_B": head(TIER_B_CAP),
            "main_C": head(TIER_C_CAP),
            "cf": head(None, CF_CAP),
            "cf_B": head(None, CF_CAP_B),
            "format": head(None, FORMAT_CAP),
            "decode": head(None, DECODE_CAP),
            "contam": contam,
            "contam_canonical": canon}


def plan_manifest(plan):
    return {"seed": SEED, "main_cap": MAIN_CAP,
            "transition_split": TRANSITION_SPLIT, "tier_b_cap": TIER_B_CAP,
            "cf_cap": CF_CAP, "cf_cap_B": CF_CAP_B,
            "format_cap": FORMAT_CAP, "decode_cap": DECODE_CAP,
            "decode_seeds": list(DECODE_SEEDS),
            "n_main": len(plan["main"]), "n_main_B": len(plan["main_B"]),
            "n_main_C": len(plan["main_C"]),
            "main_ids": [i["id"] for i in plan["main"]],
            "main_ids_B": [i["id"] for i in plan["main_B"]],
            "main_ids_C": [i["id"] for i in plan["main_C"]],
            "cf_ids": [i["id"] for i in plan["cf"]],
            "cf_ids_B": [i["id"] for i in plan["cf_B"]],
            "format_ids": [i["id"] for i in plan["format"]],
            "decode_ids": [i["id"] for i in plan["decode"]],
            "contam_ids": [i["id"] for i in plan["contam"]],
            "n_cf_variants": sum(len(i["counterfactual_variants"])
                                 for i in plan["cf"]),
            "n_cf_variants_B": sum(len(i["counterfactual_variants"])
                                   for i in plan["cf_B"])}
