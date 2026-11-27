#!/usr/bin/env python3
"""Correct the body-location phrase in PAMAP2 questions (Nov 2026 revision).

The submitted generator described every PAMAP2 window as coming from "a
chest-mounted IMU", but the windows are drawn from the hand, chest and ankle
IMUs. Ground truths are unaffected (they do not depend on the stated
location); only the question text is wrong. This script regenerates the
benchmark with source capture, recovers each item's placement, and writes
data/benchmark_rev2611/items.jsonl with corrected questions plus
changed_ids.json. data/benchmark/ is left untouched.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src"))

import rev2611_gt_validation as gv  # noqa: E402  (wraps Generator.add_*)

PHRASE = {"hand": "a wrist-worn IMU on the dominant hand",
          "chest": "a chest-mounted IMU", "ankle": "an ankle-mounted IMU"}


def main():
    ref = [json.loads(l) for l in open(os.path.join(
        ROOT, "data", "benchmark", "items.jsonl"))]
    gv.regenerate()
    changed, out = [], []
    for it in ref:
        it = dict(it)
        if it["dataset"] == "pamap2":
            pl = it["meta"].get("placement")
            if pl is None and it["id"] in gv.CAPTURE:
                _, args, _, _, _ = gv.CAPTURE[it["id"]]
                pl = args[0]["meta"].get("placement")
            if pl is None:
                raise SystemExit(f"no placement for {it['id']}")
            it["meta"] = dict(it["meta"], placement=pl)
            q = it["question"].replace("a chest-mounted IMU", PHRASE[pl])
            if q != it["question"]:
                it["question"] = q
                changed.append(it["id"])
        out.append(it)
    od = os.path.join(ROOT, "data", "benchmark_rev2611")
    os.makedirs(od, exist_ok=True)
    with open(os.path.join(od, "items.jsonl"), "w") as f:
        for it in out:
            f.write(json.dumps(it) + "\n")
    with open(os.path.join(od, "changed_ids.json"), "w") as f:
        json.dump(changed, f)
    print(len(changed), "questions corrected")


if __name__ == "__main__":
    main()
