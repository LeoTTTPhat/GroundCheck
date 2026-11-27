#!/usr/bin/env python3
"""What do models compute under the `code_agent` condition?

Reads every code_agent record in results/raw_rev2611/grid_*.jsonl (base
variant) and classifies the last code block the model ran, so that the
paper's statements about model-written code (Sec. VI, "Model-written code is
not a substitute") are counted rather than anecdotal:

  * ran_ok      at least one execution exited without error
  * no_run      the model answered without running any code
  * silent_last_run  the last execution succeeded but printed nothing
  * ignores_data     the last code never reads the signal arrays
                     (e.g. hard-coded "example" indices)
  * filter      the code smooths/filters/detrends before measuring
  * prominence  find_peaks called with a prominence argument
  * height      find_peaks called with a non-None height argument
  * over/under  sign of (prediction - reference) on wrong numeric answers

Output: results/rev2611/code_behaviour.json
"""
import collections
import glob
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILTER = re.compile(r"butter|filtfilt|sosfilt|lfilter|savgol|uniform_filter|"
                    r"gaussian_filter|convolve|rolling|medfilt|detrend|"
                    r"bandpass|highpass|lowpass|moving[_ ]?av|smooth",
                    re.I)
HEIGHT = re.compile(r"height\s*=\s*(?!None)")


def last_code(r):
    runs = r.get("code_runs") or []
    ok = [c for c in runs if c.get("status") == "ok"]
    pick = ok[-1] if ok else (runs[-1] if runs else None)
    return (pick or {}).get("code", "")


def main():
    out = {}
    for path in sorted(glob.glob(os.path.join(
            ROOT, "results", "raw_rev2611", "grid_*.jsonl"))):
        tag = os.path.basename(path)[5:-6]
        if tag.endswith("-wording"):
            continue
        rows = [json.loads(l) for l in open(path)]
        rows = [r for r in rows if r.get("backend") == "code_agent"
                and r.get("variant", "base") == "base"]
        if not rows:
            continue
        fam = collections.defaultdict(list)
        for r in rows:
            fam[r["family"]].append(r)
            fam["_all"].append(r)
        res = {}
        for f, rs in fam.items():
            n = len(rs)
            codes = [last_code(r) for r in rs]
            ran_ok = sum(any(c.get("status") == "ok"
                             for c in (r.get("code_runs") or [])) for r in rs)
            no_run = sum(not (r.get("code_runs") or []) for r in rs)
            silent = sum(bool(r.get("code_runs")) and
                         r["code_runs"][-1].get("status") == "ok" and
                         not r["code_runs"][-1].get("stdout", "").strip()
                         for r in rs)
            d = {"n": n, "acc": round(sum(r["correct"] for r in rs) / n, 3),
                 "silent_last_run": round(silent / n, 3),
                 "ignores_data": round(sum(
                     not re.search(r"\bacc\b|\bA\b|\bB\b", c)
                     for c in codes) / n, 3),
                 "ran_ok": round(ran_ok / n, 3),
                 "no_run": round(no_run / n, 3),
                 "filter": round(sum(bool(FILTER.search(c))
                                     for c in codes) / n, 3)}
            if f == "dominant_freq":
                d["spectral"] = round(sum(bool(re.search(
                    r"fft|periodogram|welch|spectrogram|csd", c, re.I))
                    for c in codes) / n, 3)
            if f == "peak_count":
                withfp = [c for c in codes if "find_peaks" in c]
                d["find_peaks"] = round(len(withfp) / n, 3)
                d["prominence"] = round(sum("prominence" in c
                                            for c in codes) / n, 3)
                d["height"] = round(sum(bool(HEIGHT.search(c))
                                        for c in codes) / n, 3)
                d["no_filter_no_prom"] = round(sum(
                    not FILTER.search(c) and "prominence" not in c
                    for c in codes) / n, 3)
            num = [r for r in rs if not r["correct"]
                   and isinstance(r.get("pred"), (int, float))
                   and isinstance(r.get("ground_truth"), (int, float))]
            if num:
                d["wrong_over"] = round(sum(r["pred"] > r["ground_truth"]
                                            for r in num) / len(num), 3)
                rel = sorted(abs(r["pred"] - r["ground_truth"])
                             / max(abs(r["ground_truth"]), 1e-9)
                             for r in num)
                d["wrong_median_rel_err"] = round(rel[len(rel) // 2], 3)
            res[f] = d
        out[tag] = res
    dst = os.path.join(ROOT, "results", "rev2611", "code_behaviour.json")
    with open(dst, "w") as f:
        json.dump(out, f, indent=1)
    for tag, res in out.items():
        print(tag)
        for f, d in sorted(res.items()):
            print("  ", f, d)


if __name__ == "__main__":
    main()
