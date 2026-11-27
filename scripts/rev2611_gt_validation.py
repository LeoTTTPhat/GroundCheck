#!/usr/bin/env python3
"""Ground-truth validity checks (November 2026 revision, reviews #63A/#63C).

Question answered: is each item's reference answer a property of the
recording, or an artifact of one DSP parameter choice?

Method. Re-run the deterministic generator (seed 0) into a scratch folder
with every add_* method wrapped so the full-rate source arrays of each item
are captured; check that the regenerated items match data/benchmark exactly;
then recompute each answer with alternative, equally reasonable algorithms
and parameters and report how often they agree within the item's tolerance.

  python3 scripts/rev2611_gt_validation.py      # -> results/rev2611/gt_validation.json

Touches nothing under data/benchmark or results/raw.
"""
import json
import os
import sys
import tempfile

import numpy as np
from scipy import signal as sps

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import dsp_tools  # noqa: E402
import evalplan  # noqa: E402
import generator  # noqa: E402
import ground_truth as gt  # noqa: E402

CAPTURE = {}


def wrap(name):
    orig = getattr(generator.Generator, name)

    def inner(self, *a, **k):
        n_before = len(self.items)
        out = orig(self, *a, **k)
        if out is not None and len(self.items) > n_before:
            CAPTURE[out["id"]] = (name, a, k, self.tfs, self.maxs)
        return out
    setattr(generator.Generator, name, inner)


for nm in ("add_peak_count", "add_dominant_freq", "add_intensity",
           "add_transition", "add_spliced_transition", "add_bout"):
    wrap(nm)


def regenerate():
    import generate_benchmark
    td = tempfile.mkdtemp(prefix="gc_regen_")
    argv = sys.argv
    sys.argv = ["generate_benchmark.py", "--seed", "0", "--out", td]
    try:
        generate_benchmark.main()
    finally:
        sys.argv = argv
    return [json.loads(l) for l in open(os.path.join(td, "items.jsonl"))]


def cropped(w, tfs, maxs, max_samples=None):
    """Full-rate samples that survive serialization (as the generator)."""
    from serialize import serialize
    acc, fs = w["acc"], w["fs"]
    _, ds, ds_fs = serialize(acc, fs, tfs, 2, max_samples or maxs)
    return acc[:int(round(ds.shape[0] * fs / ds_fs))], fs


# --------------------------------------------------------- alternatives --

def peaks_alt(acc, fs, lo=0.5, hi=5.0, prom=0.3, dist_s=None):
    mag = gt.magnitude(acc)
    filt = gt.bandpass_mag(mag, fs, lo=lo, hi=hi)
    sd = filt.std()
    if sd < 1e-12:
        return 0
    dist = max(1, int(round(fs * (dist_s if dist_s else 1.0 / hi))))
    p, _ = sps.find_peaks(filt, prominence=prom * sd, distance=dist)
    return int(len(p))


def acf_period(x, fs, lo_s=0.2, hi_s=2.5):
    x = x - x.mean()
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    if ac[0] <= 0:
        return None
    ac = ac / ac[0]
    a, b = int(lo_s * fs), min(int(hi_s * fs), len(ac) - 1)
    if b <= a + 1:
        return None
    pk, _ = sps.find_peaks(ac[a:b])
    if len(pk) == 0:
        return None
    best = pk[np.argmax(ac[a:b][pk])]
    return (a + best) / fs


def upcross_count(acc, fs):
    filt = gt.bandpass_mag(gt.magnitude(acc), fs, lo=0.5, hi=5.0)
    return int(np.sum((filt[:-1] < 0) & (filt[1:] >= 0)))


def freq_alts(acc, fs):
    mag = gt.magnitude(acc)
    x = mag - mag.mean()
    hi = min(5.0, 0.45 * 10.0)
    out = {}
    f, p = sps.periodogram(x, fs=fs)
    band = (f >= 0.3) & (f <= hi)
    out["periodogram"] = float(f[band][np.argmax(p[band])])
    for lab, nper in (("welch_2s", int(2 * fs)), ("welch_8s", int(8 * fs))):
        nper = min(len(x), max(16, nper))
        f, p = sps.welch(x, fs=fs, nperseg=nper)
        band = (f >= 0.3) & (f <= hi)
        out[lab] = float(f[band][np.argmax(p[band])])
    per = acf_period(gt.bandpass_mag(mag, fs, 0.3, 5.0), fs)
    out["autocorr"] = float(1.0 / per) if per else None
    dur = len(x) / fs
    out["zero_cross"] = upcross_count(acc, fs) / dur
    return out


def intensity_alts(a, b):
    def std_mag(x):
        return float(gt.magnitude(x).std())

    def rms_hp(x):
        hp = x - x.mean(axis=0, keepdims=True)
        return float(np.sqrt((hp ** 2).sum(axis=1).mean()))

    def jerk(x):
        return float(np.abs(np.diff(gt.magnitude(x))).mean())
    return {k: ("A" if f(a) > f(b) else "B")
            for k, f in (("std_magnitude", std_mag), ("rms_highpass", rms_hp),
                         ("mean_abs_jerk", jerk))}


def within(gtv, tol, pred):
    if pred is None:
        return False
    if tol["type"] == "absolute":
        return abs(pred - gtv) <= tol["value"]
    return abs(pred - gtv) <= max(tol["value"] * abs(gtv),
                                  tol.get("min_abs", 0.0))


def main():
    ref = {i["id"]: i for i in evalplan.load_items(
        os.path.join(ROOT, "data", "benchmark", "items.jsonl"))}
    regen = regenerate()
    mism = [i["id"] for i in regen
            if i["id"] not in ref
            or i["ground_truth"] != ref[i["id"]]["ground_truth"]
            or i["question"] != ref[i["id"]]["question"]]
    plan = evalplan.build_plan(list(ref.values()))
    main_ids = {i["id"] for i in plan["main"]}
    pool_ids = {i["id"] for i in plan["main_full"]}

    rows = []
    for iid, (name, a, k, tfs, maxs) in CAPTURE.items():
        if iid not in pool_ids:
            continue
        it = ref[iid]
        tol, gtv = it["tolerance"], it["ground_truth"]
        row = {"id": iid, "family": it["family"], "dataset": it["dataset"],
               "activity": it["meta"].get("activity"),
               "in_main": iid in main_ids}
        if name == "add_peak_count":
            acc, fs = cropped(a[0], tfs, maxs)
            dur = acc.shape[0] / fs
            alts = {"band_0.3-3Hz": peaks_alt(acc, fs, 0.3, 3.0),
                    "band_0.5-3Hz": peaks_alt(acc, fs, 0.5, 3.0),
                    "band_0.8-4Hz": peaks_alt(acc, fs, 0.8, 4.0),
                    "prom_0.15sd": peaks_alt(acc, fs, prom=0.15),
                    "prom_0.5sd": peaks_alt(acc, fs, prom=0.5),
                    "dist_0.3s": peaks_alt(acc, fs, dist_s=0.3),
                    "dist_0.4s": peaks_alt(acc, fs, dist_s=0.4)}
            fr = freq_alts(acc, fs)
            indep = {"duration_x_welch_freq": dur * gt.dominant_freq(
                         acc, fs, max_freq=4.5),
                     "upward_zero_crossings": upcross_count(acc, fs),
                     "duration_over_acf_period":
                         (dur * fr["autocorr"]) if fr["autocorr"] else None}
            row.update(gt=gtv, alts=alts, indep=indep, dur=dur,
                       cadence_spm=60.0 * gtv / dur,
                       agree={kk: within(gtv, tol, v) for kk, v in
                              {**alts, **indep}.items()})
        elif name == "add_dominant_freq":
            acc, fs = cropped(a[0], tfs, maxs)
            alts = freq_alts(acc, fs)
            row.update(gt=gtv, alts=alts,
                       agree={kk: within(gtv, tol, v)
                              for kk, v in alts.items()},
                       harmonic={kk: (v is not None and not within(gtv, tol, v)
                                      and (abs(v - 2 * gtv) <= 0.3
                                           or abs(v - gtv / 2) <= 0.3))
                                 for kk, v in alts.items()})
        elif name == "add_intensity":
            wa, wb = a[0], a[1]
            xa, _ = cropped(wa, tfs, maxs)
            xb, _ = cropped(wb, tfs, maxs)
            alts = intensity_alts(xa, xb)
            row.update(gt=gtv, alts=alts,
                       agree={kk: v == gtv for kk, v in alts.items()},
                       sma_ratio=max(gt.sma(xa), gt.sma(xb)) /
                       max(1e-12, min(gt.sma(xa), gt.sma(xb))))
        elif name == "add_transition":
            w = a[0]
            acc, fs = w["acc"], w["fs"]
            shown_cp = dsp_tools.transition_detect(acc, fs)
            meta = it["meta"]
            row.update(gt=gtv, shown_placement_cp=shown_cp,
                       label_boundary=meta.get("label_boundary_s"),
                       annotation_lag=meta.get("annotation_lag_s"),
                       agree={"shown_placement_changepoint":
                              within(gtv, tol, shown_cp),
                              "raw_label_boundary": within(
                                  gtv, tol, meta.get("label_boundary_s"))
                              if meta.get("label_boundary_s") is not None
                              else None})
        elif name == "add_spliced_transition":
            wa, wb, frac = a[0], a[1], a[2]
            fs = wa["fs"]
            n = min(wa["acc"].shape[0], wb["acc"].shape[0])
            sp = gt.splice(wa["acc"][:n], wb["acc"][:n], frac)
            cp = dsp_tools.transition_detect(sp, fs)
            row.update(gt=gtv, detected=cp,
                       agree={"energy_changepoint": within(gtv, tol, cp)})
        elif name == "add_bout":
            w_rest, w_act, frac = a[0], a[1], a[2]
            fs = w_rest["fs"]
            n = min(w_rest["acc"].shape[0], w_act["acc"].shape[0])
            n = min(n, int(round(maxs / tfs * fs)))
            rest, act = w_rest["acc"][:n], w_act["acc"][:n]
            nb = int(round(n * frac))
            s0 = (n - nb) // 2
            comp = np.concatenate([rest[:s0], act[:nb], rest[s0 + nb:]])
            det = {f"energy_bout_smooth_{s}s":
                   dsp_tools.active_bout_duration(comp, fs, smooth_s=s)
                   for s in (0.5, 1.5, 3.0)}
            row.update(gt=gtv, detected=det,
                       agree={kk: within(gtv, tol, v) for kk, v in det.items()})
        rows.append(row)

    def summarize(sel):
        out = {}
        for fam in evalplan.FAMILIES:
            fr = [r for r in sel if r["family"] == fam]
            if fam == "transition_time":
                for prov, key in (("real", "shown_placement_cp"),
                                  ("spliced", "detected")):
                    sub = [r for r in fr if key in r]
                    out[f"{fam}_{prov}"] = agree_summary(sub)
                    if prov == "real":
                        lags = [r["annotation_lag"] for r in sub
                                if r["annotation_lag"] is not None]
                        out[f"{fam}_{prov}"]["median_annotation_lag_s"] = (
                            float(np.median(lags)) if lags else None)
                        out[f"{fam}_{prov}"]["median_abs_shown_vs_gt_s"] = (
                            float(np.median([abs(r["shown_placement_cp"]
                                                 - r["gt"]) for r in sub]))
                            if sub else None)
                continue
            out[fam] = agree_summary(fr)
            if fam == "peak_count":
                byact = {}
                for r in fr:
                    byact.setdefault(r["activity"], []).append(
                        r["cadence_spm"])
                out[fam]["cadence_spm_by_activity"] = {
                    a: {"n": len(v), "median": round(float(np.median(v)), 1),
                        "p10": round(float(np.percentile(v, 10)), 1),
                        "p90": round(float(np.percentile(v, 90)), 1)}
                    for a, v in sorted(byact.items())}
                param_keys = [k for k in fr[0]["alts"]] if fr else []
                k_ok = [sum(r["agree"][k] for k in param_keys) for r in fr]
                out[fam]["items_all_param_variants_agree"] = (
                    round(float(np.mean([x == len(param_keys) for x in k_ok])),
                          3) if fr else None)
                out[fam]["items_majority_param_variants_agree"] = (
                    round(float(np.mean([x > len(param_keys) / 2
                                         for x in k_ok])), 3)
                    if fr else None)
            if fam == "dominant_freq":
                out[fam]["harmonic_disagreement_rate"] = {
                    k: round(float(np.mean([r["harmonic"][k] for r in fr])), 3)
                    for k in fr[0]["harmonic"]} if fr else {}
            if fam == "intensity_order":
                out[fam]["min_sma_ratio"] = round(min(r["sma_ratio"]
                                                      for r in fr), 3)
        return out

    res = {"regenerated_items": len(regen), "reference_items": len(ref),
           "regeneration_mismatches": mism[:20],
           "n_mismatch": len(mism),
           "main_subset": summarize([r for r in rows if r["in_main"]]),
           "full_pool": summarize(rows),
           "n_rows_main": sum(r["in_main"] for r in rows),
           "n_rows_pool": len(rows)}
    out_dir = os.path.join(ROOT, "results", "rev2611")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "gt_validation.json"), "w") as f:
        json.dump(res, f, indent=2, default=float)
    with open(os.path.join(out_dir, "gt_validation_rows.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=float) + "\n")
    print(json.dumps({k: res[k] for k in ("regenerated_items",
                                          "reference_items", "n_mismatch",
                                          "n_rows_main", "n_rows_pool")}))
    print(json.dumps(res["main_subset"], indent=1, default=float))


def agree_summary(rows):
    if not rows:
        return {"n": 0}
    keys = sorted({k for r in rows for k, v in r["agree"].items()
                   if v is not None})
    return {"n": len(rows),
            "agree_within_tolerance": {
                k: round(float(np.mean([bool(r["agree"][k]) for r in rows
                                        if r["agree"].get(k) is not None])),
                         3) for k in keys}}


if __name__ == "__main__":
    main()
