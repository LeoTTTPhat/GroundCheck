#!/usr/bin/env python3
"""Generate the GroundCheck benchmark into data/benchmark/.

Usage: python3 scripts/generate_benchmark.py [--seed 0] [--out data/benchmark]

Deterministic: all sampling is driven by --seed. Skips datasets whose raw
files are missing (UCI-HAR is mandatory). Writes items.jsonl, config.json,
stats.json.
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import loaders  # noqa: E402
from generator import Generator  # noqa: E402


def np_json(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    raise TypeError(type(o))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "benchmark"))
    ap.add_argument("--target_fs", type=float, default=10.0)
    ap.add_argument("--max_samples", type=int, default=120)
    # per-family caps (base items)
    ap.add_argument("--n_peak_ucihar", type=int, default=120)
    ap.add_argument("--n_freq_ucihar", type=int, default=120)
    ap.add_argument("--n_intensity", type=int, default=120)
    ap.add_argument("--n_peak_wisdm", type=int, default=80)
    ap.add_argument("--n_freq_wisdm", type=int, default=80)
    ap.add_argument("--n_trans_pamap2", type=int, default=120)
    ap.add_argument("--n_peak_pamap2", type=int, default=60)
    ap.add_argument("--n_freq_pamap2", type=int, default=60)
    ap.add_argument("--n_intensity_pamap2", type=int, default=60)
    ap.add_argument("--n_trans_spliced", type=int, default=60)
    ap.add_argument("--n_bout", type=int, default=80)
    ap.add_argument("--n_contam_pairs", type=int, default=60)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    g = Generator(seed=args.seed, target_fs=args.target_fs,
                  max_samples=args.max_samples)
    counts = {}
    datasets_used = []

    # ------------------------------------------------------- UCI-HAR -----
    train = loaders.load_ucihar("train")
    segs = loaders.stitch_ucihar(train, min_windows=8)
    datasets_used.append("ucihar")
    active = [s for s in segs if s["activity"] in
              ("walking", "walking_upstairs", "walking_downstairs")]
    static = [s for s in segs if s["activity"] in
              ("sitting", "standing", "laying")]
    rng.shuffle(active)
    rng.shuffle(static)

    def crop(seg, n, off=0):
        w = dict(seg)
        w["acc"] = seg["acc"][off:off + n]
        return w

    n12 = int(12.0 * 50)  # 12 s at 50 Hz

    # peak_count + dominant_freq on active segments (12 s crops)
    made_peak = made_freq = 0
    for i, seg in enumerate(active):
        if seg["acc"].shape[0] < 2 * n12 + 100:
            continue
        # two disjoint crops from the same segment: one per family
        off1 = int(rng.integers(0, seg["acc"].shape[0] - 2 * n12))
        off2 = off1 + n12 + int(rng.integers(0, min(
            50, seg["acc"].shape[0] - 2 * n12 - off1 + 1)))
        if made_peak < args.n_peak_ucihar:
            w = crop(seg, n12, off1)
            item = g.add_peak_count(w)
            if item is not None:
                made_peak += 1
                # signal_swap from a static or different-cadence segment
                pool = static if rng.random() < 0.6 else \
                    [s for s in active if s["activity"] != w["activity"]]
                if pool:
                    sw = pool[int(rng.integers(0, len(pool)))]
                    if sw["acc"].shape[0] >= n12:
                        g.add_swap_cf(item, crop(sw, n12))
        if made_freq < args.n_freq_ucihar:
            w = crop(seg, n12, off2)
            item = g.add_dominant_freq(w)
            if item is not None:
                made_freq += 1
                pool = [s for s in active
                        if s["activity"] != w["activity"]] or static
                sw = pool[int(rng.integers(0, len(pool)))]
                if sw["acc"].shape[0] >= n12:
                    g.add_swap_cf(item, crop(sw, n12))
        if made_peak >= args.n_peak_ucihar and made_freq >= args.n_freq_ucihar:
            break
    counts["ucihar_peak"] = made_peak
    counts["ucihar_freq"] = made_freq

    # intensity ordering: one active vs one static window (roles shuffled)
    made = 0
    k = 0
    while made < args.n_intensity and k < 4 * args.n_intensity:
        k += 1
        a = active[int(rng.integers(0, len(active)))]
        s = static[int(rng.integers(0, len(static)))]
        if a["acc"].shape[0] < n12 or s["acc"].shape[0] < n12:
            continue
        wa, wb = (a, s) if rng.random() < 0.5 else (s, a)
        if g.add_intensity(crop(wa, n12), crop(wb, n12)) is not None:
            made += 1
    counts["ucihar_intensity"] = made

    # spliced transitions (constructed, UCI-HAR): static -> active
    made = 0
    k = 0
    n20 = int(20.0 * 50)
    while made < args.n_trans_spliced and k < 4 * args.n_trans_spliced:
        k += 1
        a = active[int(rng.integers(0, len(active)))]
        s = static[int(rng.integers(0, len(static)))]
        if a["acc"].shape[0] < n20 or s["acc"].shape[0] < n20:
            continue
        frac = float(rng.uniform(0.25, 0.7))
        first, second = (s, a) if rng.random() < 0.7 else (a, s)
        if g.add_spliced_transition(crop(first, n20), crop(second, n20),
                                    frac) is not None:
            made += 1
    counts["ucihar_spliced_transition"] = made

    # bout duration (constructed rest|active|rest)
    made = 0
    k = 0
    while made < args.n_bout and k < 4 * args.n_bout:
        k += 1
        a = active[int(rng.integers(0, len(active)))]
        s = static[int(rng.integers(0, len(static)))]
        if a["acc"].shape[0] < n20 or s["acc"].shape[0] < n20:
            continue
        frac = float(rng.uniform(0.2, 0.55))
        if g.add_bout(crop(s, n20), crop(a, n20), frac) is not None:
            made += 1
    counts["ucihar_bout"] = made

    # contamination pairs from the TEST split (the famous benchmark windows)
    test = loaders.load_ucihar("test")
    act_test = [w for w in test if w["activity"] in
                ("walking", "walking_upstairs", "walking_downstairs")]
    rng.shuffle(act_test)
    made = 0
    for i, w in enumerate(act_test):
        if made >= args.n_contam_pairs:
            break
        fam = "peak_count" if made % 2 == 0 else "dominant_freq"
        if g.add_contamination_pair(w, fam):
            made += 1
    counts["ucihar_contam_pairs"] = made

    # --------------------------------------------------------- WISDM -----
    try:
        try:
            wis = loaders.load_wisdm(device="phone",
                                     activities=("A", "B", "C"),
                                     window_s=12.0, max_windows_per_file=3)
            wis_static = loaders.load_wisdm(device="phone",
                                            activities=("D", "E"),
                                            window_s=12.0,
                                            max_windows_per_file=2)
            datasets_used.append("wisdm_uci507")
        except FileNotFoundError:
            wis = loaders.load_wisdm_v11(
                activities=("Walking", "Jogging", "Upstairs", "Downstairs"),
                window_s=12.0, max_windows_per_key=3)
            wis_static = loaders.load_wisdm_v11(
                activities=("Sitting", "Standing"), window_s=12.0,
                max_windows_per_key=2)
            datasets_used.append("wisdm_v1.1")
        rng.shuffle(wis)
        made_p = made_f = 0
        for i, w in enumerate(wis):
            if made_p < args.n_peak_wisdm:
                item = g.add_peak_count(w)
                if item is not None:
                    made_p += 1
                    if wis_static:
                        sw = wis_static[int(rng.integers(0, len(wis_static)))]
                        g.add_swap_cf(item, sw)
                    continue
            if made_f < args.n_freq_wisdm:
                item = g.add_dominant_freq(w)
                if item is not None:
                    made_f += 1
                    pool = [x for x in wis[:200]
                            if x["activity"] != w["activity"]]
                    if pool:
                        g.add_swap_cf(item, pool[int(rng.integers(
                            0, len(pool)))])
            if made_p >= args.n_peak_wisdm and made_f >= args.n_freq_wisdm:
                break
        counts["wisdm_peak"] = made_p
        counts["wisdm_freq"] = made_f
    except FileNotFoundError as e:
        print("WISDM skipped:", e)

    # -------------------------------------------------------- PAMAP2 -----
    # Real label-derived activity transitions (exact stop-of-A boundary),
    # plus clean single-activity windows across three IMU placements
    # (hand / chest / ankle) -- a third dataset, a third body location, and
    # the only non-constructed transition items in the benchmark.
    try:
        files = loaders.pamap2_files()
        made = 0
        for fpath in files:
            if made >= args.n_trans_pamap2:
                break
            sid = int(os.path.basename(fpath)[7:10])
            tw = loaders.pamap2_consensus_transitions(
                fpath, subject=sid, window_s=20.0, seed=args.seed + sid)
            rng.shuffle(tw)
            per_file = 0
            for w in tw:
                if made >= args.n_trans_pamap2 or per_file >= 24:
                    break
                if g.add_transition(w) is not None:
                    made += 1
                    per_file += 1
        counts["pamap2_transition_real"] = made

        # PAMAP2 peak_count / dominant_freq / intensity_order
        pam_windows = []
        for fpath in files:
            sid = int(os.path.basename(fpath)[7:10])
            pam_windows.extend(loaders.pamap2_clean_windows(
                fpath, subject=sid, window_s=12.0, max_per_activity=2))
        rng.shuffle(pam_windows)
        pam_active = [w for w in pam_windows if w["activity"] in
                      ("walking", "running", "nordic_walking", "cycling",
                       "ascending_stairs", "descending_stairs",
                       "rope_jumping")]
        pam_static = [w for w in pam_windows if w["activity"] in
                      ("lying", "sitting", "standing", "ironing")]
        made_p = made_f = 0
        for i, w in enumerate(pam_active):
            if made_p < args.n_peak_pamap2:
                it = g.add_peak_count(w)
                if it is not None:
                    made_p += 1
                    pool = [x for x in pam_active
                            if x["activity"] != w["activity"]]
                    if pool:
                        g.add_swap_cf(it, pool[int(rng.integers(
                            0, len(pool)))])
                    continue
            if made_f < args.n_freq_pamap2:
                it = g.add_dominant_freq(w)
                if it is not None:
                    made_f += 1
                    pool = [x for x in pam_active
                            if x["activity"] != w["activity"]]
                    if pool:
                        g.add_swap_cf(it, pool[int(rng.integers(
                            0, len(pool)))])
            if made_p >= args.n_peak_pamap2 and made_f >= args.n_freq_pamap2:
                break
        counts["pamap2_peak"] = made_p
        counts["pamap2_freq"] = made_f

        made_i = 0
        for k in range(min(len(pam_active), len(pam_static))):
            if made_i >= args.n_intensity_pamap2:
                break
            if g.add_intensity(pam_active[k], pam_static[k]) is not None:
                made_i += 1
        counts["pamap2_intensity"] = made_i
        datasets_used.append("pamap2")
    except (FileNotFoundError, ImportError) as e:
        print("PAMAP2 skipped:", e)

    # ------------------------------------------- alt serialization pass --
    # Every item also carries magnitude-only / time-indexed / coarse
    # renderings of the SAME serialized window, so the format-robustness
    # conditions add no information -- only a different text form.
    import serialize as _ser  # noqa: E402
    for it in g.items:
        for k, v in _ser.alt_formats_from_text(
                it["serialized_signal"]).items():
            it["serialized_" + k] = v

    # ---------------------------------------------------------- write ----
    os.makedirs(args.out, exist_ok=True)
    items_path = os.path.join(args.out, "items.jsonl")
    with open(items_path, "w") as f:
        for it in g.items:
            f.write(json.dumps(it, default=np_json) + "\n")

    n_cf = sum(len(it["counterfactual_variants"]) for it in g.items)
    fam_counts = {}
    for it in g.items:
        fam_counts[it["family"]] = fam_counts.get(it["family"], 0) + 1
    ds_counts, real_trans = {}, 0
    for it in g.items:
        ds_counts[it["dataset"]] = ds_counts.get(it["dataset"], 0) + 1
        if it["family"] == "transition_time" and it["meta"].get(
                "real_transition"):
            real_trans += 1
    stats = {"n_items": len(g.items), "n_counterfactual_variants": n_cf,
             "per_family": fam_counts, "per_source": counts,
             "per_dataset": ds_counts,
             "real_transition_items": real_trans,
             "datasets_used": datasets_used}
    with open(os.path.join(args.out, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(json.dumps(stats, indent=2))
    print("wrote", items_path)


if __name__ == "__main__":
    main()
