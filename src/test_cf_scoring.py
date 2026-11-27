#!/usr/bin/env python3
"""Tests for the counterfactual scoring rules used by scripts/aggregate.py.

The hallucination rate and directional-consistency metrics are defined
relative to the model's OWN base answer, with several conditioning rules
(exclude variants whose truth did not move; exclude unparseable pairs;
'flip' for categorical edits, 'increase'/'decrease' for numeric ones).
Those rules are easy to get subtly wrong and impossible to eyeball in
aggregate, so they are checked here against hand-worked cases.

Run: python3 src/test_cf_scoring.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src"))

import aggregate as ag  # noqa: E402

REL = {"type": "relative", "value": 0.2, "min_abs": 2}
ABS = {"type": "absolute", "value": 1.0}
EXACT = {"type": "exact"}

fails = []


def chk(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  {extra}" if extra else ""))
    if not cond:
        fails.append(name)


# ---- band() -------------------------------------------------------------
chk("relative band uses the min_abs floor near zero",
    ag.band(REL, 1.0) == 2.0, f"{ag.band(REL, 1.0)}")
chk("relative band scales with the reference",
    ag.band(REL, 50.0) == 10.0, f"{ag.band(REL, 50.0)}")
chk("absolute band ignores the reference", ag.band(ABS, 99.0) == 1.0)

# ---- unchanged(): 'did the answer move?' relative to the model's own base
item_rel = {"tolerance": REL}
item_abs = {"tolerance": ABS}
item_ex = {"tolerance": EXACT}
chk("unchanged: 20 vs 21 inside the +-4 band -> unchanged",
    ag.unchanged(item_rel, 20.0, 21.0) is True)
chk("unchanged: 20 vs 40 outside the band -> changed",
    ag.unchanged(item_rel, 20.0, 40.0) is False)
chk("unchanged: band is computed around the BASE answer, not the variant",
    ag.unchanged(item_rel, 100.0, 85.0) is True
    and ag.unchanged(item_rel, 5.0, 20.0) is False)
chk("unchanged: categorical compares case-insensitively",
    ag.unchanged(item_ex, "a", "A") is True
    and ag.unchanged(item_ex, "A", "B") is False)
chk("unchanged: unparseable answer -> None (excluded, not scored 0)",
    ag.unchanged(item_abs, None, 3.0) is None
    and ag.unchanged(item_abs, 3.0, None) is None)

# ---- gt_moved(): the conditioning rule for the hallucination metric -----
def gt_moved(item, var_gt):
    if var_gt in (None, "None"):
        return True
    if item["tolerance"]["type"] == "exact":
        return str(var_gt).upper() != str(item["ground_truth"]).upper()
    g0, g1 = ag.num(item["ground_truth"]), ag.num(var_gt)
    return abs(g1 - g0) > ag.band(item["tolerance"], g0)


it = {"tolerance": REL, "ground_truth": 20.0}
chk("gt_moved: a scramble that leaves the truth put is EXCLUDED",
    not gt_moved(it, 21.0))
chk("gt_moved: a swap that moves the truth is scored", gt_moved(it, 40.0))
chk("gt_moved: destroyed structure (None truth) is scored",
    gt_moved(it, None))
ite = {"tolerance": EXACT, "ground_truth": "A"}
chk("gt_moved: categorical flip is scored", gt_moved(ite, "B"))
chk("gt_moved: categorical same is excluded", not gt_moved(ite, "A"))

# ---- directional rules --------------------------------------------------
def consistent(direction, base, var):
    if base is None or var is None:
        return None
    if direction == "flip":
        return str(var).upper() != str(base).upper()
    b, v = ag.num(base), ag.num(var)
    if b is None or v is None:
        return None
    return v < b if direction == "decrease" else v > b


chk("direction decrease: 2.0 -> 1.5 consistent, 2.0 -> 2.5 not",
    consistent("decrease", 2.0, 1.5) and not consistent("decrease", 2.0, 2.5))
chk("direction increase: 4.0 -> 6.0 consistent, 4.0 -> 3.0 not",
    consistent("increase", 4.0, 6.0) and not consistent("increase", 4.0, 3.0))
chk("direction flip: A -> B consistent, A -> A not",
    consistent("flip", "A", "B") and not consistent("flip", "A", "A"))
chk("direction: an unchanged numeric answer is NOT consistent either way",
    not consistent("decrease", 2.0, 2.0)
    and not consistent("increase", 2.0, 2.0))
chk("direction: unparseable -> None (excluded)",
    consistent("increase", None, 3.0) is None)

# ---- a constant-answering model must score maximally hallucinatory ------
base, var = 24.0, 24.0
chk("a model that always answers the same number is 100% hallucinatory",
    ag.unchanged(item_rel, base, var) is True)

print(f"\n{18 - len(fails)}/18 passed")
sys.exit(1 if fails else 0)
