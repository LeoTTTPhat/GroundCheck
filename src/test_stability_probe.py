#!/usr/bin/env python3
"""Negative control for the numerical-stability guard.

The audit's dependent variable is "does the model respond to a long numeric
prompt".  On MPS, reduced-precision inference degenerates into repeated
punctuation on exactly those prompts, which is indistinguishable from the
effect being measured.  A guard is only worth anything if it detects that
failure, so this test asserts both directions on a configuration we have
independently reproduced (Qwen2.5-1.5B-Instruct: bfloat16 must be rejected,
float32 must be accepted).

What we can and cannot claim about the failure
----------------------------------------------
Reproduced repeatedly:
  * bf16/fp16 degenerate on ~2k-token serialized items where float32 does not;
  * once a process has produced one degenerate reply, subsequent long-prompt
    replies in that process keep degenerating (it is sticky);
  * the production-sequence probe (real items, real token caps) rejects the
    bad configuration, while float32 passes.

NOT reproduced, and therefore NOT claimed anywhere:
  * a clean trigger rule.  Earlier we believed the failure required a
    preceding short prompt and a *real* (not synthetic) serialized item;
    neither survived testing -- a synthetic 2.4k-token prompt degenerated
    too, and a long-prompt-first sequence sometimes degenerates.  The
    onset is intermittent and we cannot predict it.

That is exactly why the load-time probe is necessary but NOT sufficient, and
why every recorded run carries an `is_degenerate` flag that the runner treats
as fatal.  A guard that only ran at load time would be trusting a trigger
rule we have shown ourselves unable to state.

Run: python3 src/test_stability_probe.py     (needs the benchmark + MPS)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import evalplan  # noqa: E402
import harness  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def _probe_items():
    items = evalplan.load_items(
        os.path.join(ROOT, "data", "benchmark", "items.jsonl"))
    plan = evalplan.build_plan(items)
    return [next(i for i in plan["main"] if i["family"] == f)
            for f in evalplan.FAMILIES]


def main():
    import torch
    probe = _probe_items()
    fails = []

    def check(name, cond):
        print(("PASS " if cond else "FAIL ") + name)
        if not cond:
            fails.append(name)

    # --- 1. the guard rejects the known-bad configuration ----------------
    llm = harness.HFLLM(MODEL, dtype="bfloat16", stability_check=False)
    st_bf16 = llm.verify_on(probe)
    check("bfloat16 is REJECTED by the production-sequence probe",
          not st_bf16["ok"])
    check("the rejected response is flagged degenerate",
          harness.is_degenerate(st_bf16["probe_response"]))

    # --- 2. the failure is sticky within a process ----------------------
    r1 = harness.run_with_signal(llm, probe[0])
    r2 = harness.run_with_signal(llm, probe[1])
    check("once degenerate, later long-prompt replies stay degenerate",
          r1["degenerate"] and r2["degenerate"])
    check("degenerate replies are unparseable, i.e. they would show up as "
          "'the model ignored the signal'",
          r1["pred"] is None and r2["pred"] is None)
    del llm
    torch.mps.empty_cache()

    # --- 3. float32 is accepted and answers cleanly ---------------------
    llm3 = harness.HFLLM(MODEL, dtype="float32", stability_check=False)
    st_fp32 = llm3.verify_on(probe)
    check("float32 is ACCEPTED by the production-sequence probe",
          st_fp32["ok"])
    ok = True
    for it in probe:
        r = harness.run_with_signal(llm3, it)
        ok = ok and (not r["degenerate"]) and r["pred"] is not None
    check("float32 answers every probe item cleanly and parseably", ok)
    del llm3
    torch.mps.empty_cache()

    n = 6
    print(f"\n{n - len(fails)}/{n} passed")
    if fails:
        print("FAILED:", "; ".join(fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
