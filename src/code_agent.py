"""`code_agent` condition (November 2026 revision).

The model writes and runs its own Python over the window, with numpy and
scipy available and no GroundCheck toolbox. This is the code-generation route
of PHIA-style wearable agents: the model must choose the algorithm and its
parameters itself, instead of calling a verified routine. It answers the
reviewer question of whether a capable agent needs supplied DSP tools.

Protocol (identical for every model):
  * the prompt is the item question plus the header line(s) of the
    serialized signal (sampling rate, duration, sample count) and the first
    five samples of each axis; the full window is NOT pasted into the prompt,
    because in a smoke test Qwen3-8B spent its budget copying the samples
    into code instead of using the preloaded arrays (PHIA-style agents
    likewise see a schema and preview, not the raw table);
  * the parsed arrays are preloaded: single-window items get ``acc`` (N x 3,
    columns x, y, z) and ``fs``; two-window items get ``A``, ``B`` and ``fs``.
    They are parsed from the serialized text, exactly as the toolbox and the
    dsp_oracle see them, so no condition receives more information;
  * the model replies with one fenced python block, which runs in a separate
    ``python3 -E`` process (temporary directory holding only the script and
    data) with a 20 s timeout; stdout/stderr (truncated) come back as the
    next user turn; at most 3 executions;
  * as in a notebook-style code interpreter, the value of a final bare
    expression is printed (v2; v1 printed only explicit output, and models
    that ended with e.g. ``num_peaks`` saw empty output and guessed);
  * a reply with a code block is executed even if it also states an answer
    (same precedence rule as the JSON tool loop);
  * the answer is the 'Final answer:' line of the first reply without code.
"""
import ast
import json
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

import harness

CODE_SYSTEM = (
    "You are an expert wearable-sensor signal analyst who can run Python 3. "
    "The signal shown below is already loaded as numpy arrays. Single-window "
    "recordings: `acc` (N x 3 array, columns x, y, z) and `fs` (sampling "
    "rate in Hz). Two-window questions: `A` and `B` (each N x 3) and `fs`. "
    "`np` (numpy), `scipy` and `sps` (scipy.signal) are imported. Use these "
    "variables; never copy sample values into your code. To run "
    "code, reply with ONE fenced ```python code block that prints what you "
    "need; you will see its output (a final bare expression is also "
    "displayed, as in a notebook). You may run code at most 3 times. When "
    "you know the answer, reply without code and end with the exact line "
    "'Final answer: <value>'.")

PRELUDE = (
    "import numpy as np, scipy, scipy.signal as sps, json, sys\n"
    "_d = np.load(sys.argv[1])\n"
    "fs = float(_d['fs'])\n"
    "acc = _d['acc'] if 'acc' in _d.files else None\n"
    "A = _d['A'] if 'A' in _d.files else None\n"
    "B = _d['B'] if 'B' in _d.files else None\n")

CODE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def _echo_last_expr(code):
    """Notebook semantics: print repr() of a final top-level expression."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code                     # let Python report the error
    if not tree.body or not isinstance(tree.body[-1], ast.Expr):
        return code
    last = tree.body[-1]
    if last.col_offset != 0:
        return code
    v = last.value
    if (isinstance(v, ast.Call) and isinstance(v.func, ast.Name)
            and v.func.id == "print"):
        return code
    seg = ast.get_source_segment(code, v)
    if seg is None:
        return code
    lines = code.splitlines()
    head = lines[:last.lineno - 1]
    if len(tree.body) > 1 and tree.body[-2].end_lineno >= last.lineno:
        return code                     # shares a line with the previous one
    return "\n".join(head + ["__gc_r = (" + seg + ")",
                             "if __gc_r is not None:",
                             "    print(repr(__gc_r))"]) + "\n"


def _run_code(code, signals, fs, timeout=20):
    with tempfile.TemporaryDirectory(prefix="gc_code_") as td:
        arrays = {"fs": np.array(fs)}
        if "B" in signals:
            arrays.update(A=signals["A"], B=signals["B"])
        else:
            arrays["acc"] = signals["A"]
        data = os.path.join(td, "sig.npz")
        np.savez(data, **arrays)
        script = os.path.join(td, "main.py")
        with open(script, "w") as f:
            f.write(PRELUDE + "\n" + _echo_last_expr(code))
        try:
            p = subprocess.run([sys.executable, "-E", script, data], cwd=td,
                               capture_output=True, text=True,
                               timeout=timeout)
            out = p.stdout[-1500:]
            err = p.stderr[-800:]
            status = "ok" if p.returncode == 0 else f"exit {p.returncode}"
        except subprocess.TimeoutExpired:
            out, err, status = "", "", "timeout"
    return {"status": status, "stdout": out, "stderr": err}


def preview(sig_text, k=5):
    """Header lines plus the first k samples of each axis line."""
    out = []
    for line in sig_text.splitlines():
        m = re.match(r"^(\s*[a-z]+:)\s*(.+)$", line)
        if m and re.match(r"^-?[\d.]", m.group(2).strip()):
            vals = m.group(2).split()
            out.append(f"{m.group(1)} {' '.join(vals[:k])} ... "
                       f"({len(vals)} values)")
        else:
            out.append(line)
    return "\n".join(out)


def run_code_agent(llm, item, variant=None, max_runs=3,
                   max_new_tokens=384):
    sig_text = (variant["serialized_signal"] if variant
                else item["serialized_signal"])
    signals, fs = harness.split_windows(sig_text)
    q = item["question"] + "\n\nSignal (loaded in Python; preview only):\n" \
        + preview(sig_text)
    messages = [{"role": "system", "content": CODE_SYSTEM},
                {"role": "user", "content": q}]
    tot_in = tot_gen = 0
    tot_dt = 0.0
    runs = []
    text = ""
    for _ in range(max_runs + 1):
        text, n_in, n_gen, dt = llm.chat(messages,
                                         max_new_tokens=max_new_tokens)
        tot_in += n_in
        tot_gen += n_gen
        tot_dt += dt
        m = CODE_RE.search(text)
        if m is None or len(runs) >= max_runs:
            break
        res = _run_code(m.group(1), signals, fs)
        runs.append({"code": m.group(1)[:4000], **res})
        feedback = (f"Execution {res['status']}.\nstdout:\n{res['stdout']}"
                    + (f"\nstderr:\n{res['stderr']}" if res["stderr"] else ""))
        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user", "content": feedback})
    if "Final answer" not in text and (CODE_RE.search(text)
                                       or text.count("```") % 2 == 1):
        pred = None          # still writing code (or cut off inside a code
        #                      block): no answer was given; never scrape
        #                      numbers out of code
    else:
        pred = harness.extract_answer(text, item["family"])
    r = harness._result("code_agent", item, text, pred, tot_in, tot_gen,
                        tot_dt, variant)
    r["code_runs"] = runs
    return r


if __name__ == "__main__":       # tiny self-test of the sandbox
    sig = {"A": np.random.default_rng(0).normal(size=(50, 3))}
    print(json.dumps(_run_code("print(acc.shape, fs)", sig, 10.0)))
    print(json.dumps(_run_code("n = acc.shape[0]\nn", sig, 10.0)))
    print(json.dumps(_run_code("x = (1,\n     2)\nx[0] + \\\n  x[1]", sig, 10.0)))
    print(json.dumps(_run_code("print('a')", sig, 10.0)))
    print(json.dumps(_run_code("acc.sum(); fs", sig, 10.0)))
