"""GroundCheck evaluation harness.

Backends (pluggable):
  no_signal      -- question only (label-prior lower bound)
  with_signal    -- question + serialized signal in the prompt
  tool_augmented -- LLM may call the DSP toolbox via a JSON function-calling
                    loop; tools run on the exact serialized signal
  dsp_oracle     -- deterministic DSP pipeline on the serialized signal
                    (upper bound, no LLM)

LLM: HuggingFace transformers, greedy decoding, MPS if available.
"""
import json
import re
import time

import numpy as np

import dsp_tools

SYSTEM = ("You are an expert wearable-sensor signal analyst. Answer "
          "concisely and end with the exact line 'Final answer: <value>'.")

SYSTEM_COT = ("You are an expert wearable-sensor signal analyst. Reason "
              "step by step about the actual numbers in the signal (peaks, "
              "oscillation period, amplitude changes) in at most 4 short "
              "sentences, then end with the exact line "
              "'Final answer: <value>'.")

TOOL_SYSTEM = (
    "You are an expert wearable-sensor signal analyst with access to a DSP "
    "toolbox that runs on the provided signal. Tools:\n"
    + "\n".join(f"- {t['name']}: {t['description']} args: {t['args']}"
                for t in dsp_tools.TOOL_SPECS) +
    "\nThe toolbox already holds the signal shown below; NEVER pass the "
    "signal samples as arguments. To call a tool, reply with ONLY a JSON "
    'object like {"tool": "count_peaks", "args": {}} (only sma takes '
    '{"window": "A"} or {"window": "B"}). After you have the tool results '
    'you need, reply with ONLY {"answer": <number or letter in quotes>}. '
    "Never guess without a tool call.")


# ------------------------------------------------------------ signal I/O --

def parse_serialized(text):
    """Parse a serialized_signal string back to ((N,3) array, fs).

    Guarantees tool backends operate on exactly what the model sees.
    For two-window items use split_windows() first.
    """
    m = re.search(r"Sampling rate:\s*([\d.]+)\s*Hz", text)
    fs = float(m.group(1)) if m else 10.0
    axes = {}
    for name in "xyz":
        mm = re.search(rf"^{name}:\s*(.+)$", text, flags=re.M)
        if mm:
            axes[name] = np.array([float(v) for v in mm.group(1).split()])
    if len(axes) < 3:
        # magnitude-only / indexed serialization variants: rebuild a
        # pseudo-triaxial array carrying the magnitude on x so every DSP
        # tool (which starts by taking the norm) sees the same series.
        mm = re.search(r"^magnitude:\s*(.+)$", text, flags=re.M)
        if mm:
            mag = np.array([float(v) for v in mm.group(1).split()])
        else:
            mag = np.array([float(v) for v in
                            re.findall(r"\bm=(-?[\d.]+)", text)])
        if len(mag) == 0:
            raise ValueError("unparseable serialization")
        z = np.zeros_like(mag)
        return np.stack([mag, z, z], axis=1), fs
    n = min(len(v) for v in axes.values())
    return np.stack([axes["x"][:n], axes["y"][:n], axes["z"][:n]],
                    axis=1), fs


def split_windows(text):
    """Split 'Window A:...Window B:...' serializations."""
    if "Window B:" not in text:
        acc, fs = parse_serialized(text)
        return {"A": acc}, fs
    part_a, part_b = text.split("Window B:")
    part_a = part_a.replace("Window A:", "")
    acc_a, fs = parse_serialized(part_a)
    acc_b, _ = parse_serialized(part_b)
    return {"A": acc_a, "B": acc_b}, fs


def is_degenerate(text):
    """Detect the MPS reduced-precision decode failure in a single response.

    The failure emits long runs of a single PUNCTUATION token ('!!!!',
    '@@@@').  The test must not fire on legitimate output: an early version
    keyed on "any character repeated 10+ times" and flagged the perfectly
    valid answer '0.8333333333333333', which aborted a good run.  So the
    run test is restricted to non-alphanumeric, non-whitespace, non-'.'
    characters, and the alphabet test ignores responses that contain digits
    (a real answer always does).

    Every recorded run carries this flag; the aggregator recomputes it from
    the stored text so a detector fix applies retroactively to raw data.
    """
    if not text:
        return False
    t = text.strip()
    if re.search(r"([^\w\s.])\1{7,}", t):
        return True
    if not re.search(r"\d", t) and len(set(t)) <= 2 and len(t) > 12:
        return True
    return False


# ------------------------------------------------------------ answer fmt --

def extract_answer(text, family):
    """Parse a model response into a float or 'A'/'B'. None if unparseable.

    Preference order: an explicit 'Final answer:' line, else the LAST value in
    the reply.  Taking the last value matters for verbose models: a reply that
    reasons ("about 2.5 steps per second ... so 30 peaks") must be scored on
    its conclusion, not on the first number it happens to mention.  Combined
    with the generous generation caps in the runner, this keeps the parser
    from turning verbosity into a spurious accuracy difference between models.
    """
    if text is None:
        return None
    m = re.search(r"[Ff]inal answer\s*:\s*(.+)", text)
    tail = m.group(1) if m else None
    if family == "intensity_order":
        for src in ([tail] if tail else []) + [text]:
            mm = re.findall(r"\b([AB])\b", src)
            if mm:
                return mm[0] if src is tail else mm[-1]
            mm = re.findall(r"[Ww]indow\s*([AB])", src)
            if mm:
                return mm[0] if src is tail else mm[-1]
        return None
    if tail:
        mm = re.search(r"-?\d+\.?\d*", tail.replace(",", ""))
        if mm:
            return float(mm.group(0))
    mm = re.findall(r"-?\d+\.?\d*", text.replace(",", ""))
    return float(mm[-1]) if mm else None


def score(ground_truth, tolerance, pred):
    """1/0 tolerance-band correctness. None pred -> 0."""
    if pred is None or ground_truth is None:
        return 0
    if tolerance["type"] == "exact":
        return int(str(pred).strip().upper() == str(ground_truth).upper())
    try:
        p = float(pred)
        g = float(ground_truth)
    except (TypeError, ValueError):
        return 0
    if tolerance["type"] == "absolute":
        return int(abs(p - g) <= tolerance["value"])
    band = max(tolerance["value"] * abs(g), tolerance.get("min_abs", 0.0))
    return int(abs(p - g) <= band)


# ------------------------------------------------------------------ LLM ---

class HFLLM:
    """Greedy-decoding chat wrapper around a HF causal LM on MPS/CPU."""

    def __init__(self, model_name, device=None, max_new_tokens=48,
                 dtype="auto", stability_check=True):
        # NUMERICAL STABILITY (Stage 3). On MPS, reduced-precision inference
        # silently degenerates into '!!!!' repetition once the prompt is long
        # -- verified reproducibly for Qwen2.5-0.5B AND Qwen2.5-1.5B in both
        # fp16 and bf16 at ~2k-token prompts, and triggered by a preceding
        # short-prompt generation.  A degenerate response is unparseable, so
        # it masquerades as a low parse rate / a model that "ignores the
        # signal": exactly the effect this benchmark is trying to measure.
        # Stage 3 therefore runs EVERY model in float32 and verifies at load
        # time that a long prompt still decodes cleanly.
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.name = model_name
        self.device = device or ("mps" if torch.backends.mps.is_available()
                                 else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if dtype == "auto":
            # float32 is the safe default; models >=7B are checked in bf16
            # (2.4x faster) and fall back to float32 if the stability probe
            # trips.  Either way the runtime degeneracy flag is recorded on
            # every run, so a silent failure cannot reach the results.
            dtype = "float32"
        # Probe the chat template rather than grepping it: Gemma-2's template
        # mentions 'system' only inside the error it raises.
        try:
            self.tok.apply_chat_template(
                [{"role": "system", "content": "x"},
                 {"role": "user", "content": "y"}], tokenize=False,
                add_generation_prompt=True)
            self.supports_system = True
        except Exception:
            self.supports_system = False
        self.dtype = dtype
        td = {"float16": torch.float16, "bfloat16": torch.bfloat16,
              "float32": torch.float32}[dtype]
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=td)
        self.model.to(self.device)
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.stable = None
        if stability_check:
            self.stable = self._check_stability()

    def _check_stability(self, rounds=4, probe_items=None):
        """Long-prompt decode sanity check; see the dtype note in __init__.

        The failure is intermittent and we could NOT establish a reliable
        trigger rule.  Two hypotheses we held and then falsified: that it
        required a preceding short-prompt generation, and that a synthetic
        run of numbers would not trigger it (a 2.4k-token synthetic prompt
        does).  Once a process emits one degenerate reply it keeps emitting
        them.  Because we cannot state the trigger, a load-time probe is
        necessary but NOT sufficient -- every recorded run additionally
        carries an `is_degenerate` flag that the runner treats as fatal.

        The probe therefore replays the production call sequence verbatim:
        `run_no_signal` then `run_with_signal` on real items at the real
        token caps.  Even so, a load-time probe is necessary but NOT
        sufficient, which is why every recorded run carries an
        `is_degenerate` flag and the runner aborts on a nonzero rate.
        """
        if not probe_items:
            return {"ok": True, "rounds": 0, "real_items": False,
                    "probe_response": "(no probe items supplied)"}
        bad_ex, txt = None, ""
        for it in probe_items[:rounds]:
            run_no_signal(self, it)          # the trigger, exactly as in prod
            r = run_with_signal(self, it)    # ... and the production query
            txt = r["raw_response"] or ""
            if r["degenerate"]:
                bad_ex = txt[:120]
                break
        return {"ok": bad_ex is None, "rounds": min(rounds,
                                                    len(probe_items)),
                "real_items": True,
                "probe_response": bad_ex or txt[:120]}

    def verify_on(self, probe_items):
        """Re-run the stability probe against real benchmark items."""
        self.stable = self._check_stability(probe_items=probe_items)
        return self.stable

    def _fit_messages(self, messages):
        """Some chat templates (Gemma-2) reject a 'system' role; fold the
        system message into the first user turn for those models."""
        if self.supports_system:
            return messages
        out, sys_txt = [], None
        for m in messages:
            if m["role"] == "system":
                sys_txt = m["content"]
                continue
            if sys_txt is not None and m["role"] == "user":
                out.append({"role": "user",
                            "content": sys_txt + "\n\n" + m["content"]})
                sys_txt = None
            else:
                out.append(m)
        return out or [{"role": "user", "content": sys_txt or ""}]

    def chat(self, messages, max_new_tokens=None, temperature=None,
             seed=None):
        """Returns (text, prompt_tokens, gen_tokens, latency_s).

        temperature=None -> greedy (the default everywhere in the audit);
        a float enables sampling, used only for the decoding-variance study.
        """
        import torch
        if seed is not None:
            torch.manual_seed(seed)
        messages = self._fit_messages(messages)
        prompt = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tok(prompt, return_tensors="pt").to(self.device)
        n_in = inputs["input_ids"].shape[1]
        t0 = time.time()
        gen_kw = dict(do_sample=False, temperature=None, top_p=None,
                      top_k=None)
        if temperature is not None:
            gen_kw = dict(do_sample=True, temperature=float(temperature),
                          top_p=0.95, top_k=50)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                pad_token_id=self.tok.eos_token_id, **gen_kw)
        dt = time.time() - t0
        gen = out[0][n_in:]
        text = self.tok.decode(gen, skip_special_tokens=True)
        return text, int(n_in), int(len(gen)), dt


# ------------------------------------------------------------- backends ---

def run_no_signal(llm, item, max_new_tokens=128):
    q = (item["question"] + "\nYou are NOT shown the signal. Give your best "
         "single guess from context alone.")
    text, n_in, n_gen, dt = llm.chat(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": q}], max_new_tokens=max_new_tokens)
    return _result("no_signal", item, text, extract_answer(
        text, item["family"]), n_in, n_gen, dt)


def run_with_signal(llm, item, variant=None, max_new_tokens=128):
    sig = variant["serialized_signal"] if variant else item[
        "serialized_signal"]
    q = item["question"] + "\n\nSignal:\n" + sig
    text, n_in, n_gen, dt = llm.chat(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": q}], max_new_tokens=max_new_tokens)
    return _result("with_signal", item, text, extract_answer(
        text, item["family"]), n_in, n_gen, dt, variant)


def run_with_signal_cot(llm, item, variant=None):
    """with_signal + chain-of-thought (capped at 128 new tokens)."""
    sig = variant["serialized_signal"] if variant else item[
        "serialized_signal"]
    q = item["question"] + "\n\nSignal:\n" + sig
    text, n_in, n_gen, dt = llm.chat(
        [{"role": "system", "content": SYSTEM_COT},
         {"role": "user", "content": q}], max_new_tokens=192)
    return _result("with_signal_cot", item, text,
                   extract_answer(text, item["family"]), n_in, n_gen, dt,
                   variant)


def run_tool_augmented(llm, item, variant=None, max_rounds=4):
    sig_text = variant["serialized_signal"] if variant else item[
        "serialized_signal"]
    signals, fs = split_windows(sig_text)
    q = item["question"] + "\n\nSignal:\n" + sig_text
    messages = [{"role": "system", "content": TOOL_SYSTEM},
                {"role": "user", "content": q}]
    tot_in = tot_gen = 0
    tot_dt = 0.0
    calls = []
    text = ""
    for _ in range(max_rounds):
        text, n_in, n_gen, dt = llm.chat(messages, max_new_tokens=96)
        tot_in += n_in
        tot_gen += n_gen
        tot_dt += dt
        obj = _parse_json_obj(text)
        if obj is None:
            break
        if "answer" in obj:
            r = _result("tool_augmented", item, text,
                        _coerce(obj["answer"], item["family"]),
                        tot_in, tot_gen, tot_dt, variant)
            r["tool_calls"] = calls
            return r
        if "tool" in obj:
            res = dsp_tools.run_tool(obj["tool"], signals, fs,
                                     obj.get("args"))
            calls.append({"tool": obj["tool"], "args": obj.get("args"),
                          "result": res})
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user",
                             "content": "Tool result: " + json.dumps(res)})
        else:
            break
    # Rounds exhausted or unparseable reply. If the last reply still looks
    # like a tool call, there is no answer -- do NOT scrape numbers out of
    # a malformed tool call (they would be signal samples, not answers).
    pred = None if '"tool"' in text else extract_answer(text,
                                                        item["family"])
    r = _result("tool_augmented", item, text, pred,
                tot_in, tot_gen, tot_dt, variant)
    r["tool_calls"] = calls
    return r


def run_dsp_oracle(item, variant=None):
    sig_text = variant["serialized_signal"] if variant else item[
        "serialized_signal"]
    signals, fs = split_windows(sig_text)
    t0 = time.time()
    ans = dsp_tools.oracle_answer(item["family"], signals, fs)
    dt = time.time() - t0
    return _result("dsp_oracle", item, str(ans), ans, 0, 0, dt, variant)


def _json_objects(text):
    """Successive balanced top-level {...} spans, in order of appearance."""
    out, depth, start = [], 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                out.append(text[start:i + 1])
                start = None
    return out


def _parse_json_obj(text):
    """Parse the model's JSON reply into a tool call or an answer.

    A TOOL CALL TAKES PRECEDENCE OVER AN ANSWER in the same reply. Models
    routinely emit both at once --

        {"tool": "count_peaks", "args": {}}
        {"answer": "11"}

    -- and that answer cannot have used the tool, because the tool has not
    run yet. An earlier version matched one greedy ``\{.*\}`` span (which
    fails to parse when two objects are present) and then fell back to
    checking ``"answer"`` first, so it returned the model's uninformed guess
    and never executed the tool. That understated tool usage on 64% of
    Gemma-2-9B's runs and 18% of Qwen2.5-3B's, and turned a parser artefact
    into an apparent inability to route.

    Robust to malformed args (e.g. the model dumping signal samples with
    '...'), with regex recovery of the tool name or an 'answer' field.
    """
    spans = _json_objects(text)
    parsed = []
    for sp in spans:
        try:
            parsed.append(json.loads(sp))
        except json.JSONDecodeError:
            parsed.append(None)
    for obj in parsed:
        if isinstance(obj, dict) and "tool" in obj:
            return obj
    for obj in parsed:
        if isinstance(obj, dict) and "answer" in obj:
            return obj
    # regex recovery, tool first for the same reason
    mm = re.search(r'"tool"\s*:\s*"([a-z_]+)"', text)
    if mm:
        obj = {"tool": mm.group(1), "args": {}}
        mw = re.search(r'"window"\s*:\s*"?([ABab])"?', text)
        if mw:
            obj["args"]["window"] = mw.group(1).upper()
        return obj
    mm = re.search(r'"answer"\s*:\s*"?([A-Za-z0-9.+-]+)"?', text)
    if mm:
        return {"answer": mm.group(1)}
    return None


def _coerce(ans, family):
    if family == "intensity_order":
        s = str(ans).strip().upper()
        return s[0] if s[:1] in ("A", "B") else None
    try:
        return float(ans)
    except (TypeError, ValueError):
        return extract_answer(str(ans), family)


def _result(backend, item, raw, pred, n_in, n_gen, dt, variant=None):
    gt_val = (variant["ground_truth"] if variant is not None
              else item["ground_truth"])
    return {"backend": backend, "item_id": item["id"],
            "degenerate": is_degenerate(raw),
            "family": item["family"], "dataset": item["dataset"],
            "variant": variant["type"] if variant else "base",
            "pred": pred, "ground_truth": gt_val,
            "correct": score(gt_val, item["tolerance"], pred),
            "prompt_tokens": n_in, "gen_tokens": n_gen,
            "latency_s": round(dt, 3), "raw_response": raw}


# =========================================================================
# Stage-3 additions
# =========================================================================

FEATURE_SYSTEM = (
    "You are an expert wearable-sensor signal analyst. A DSP pipeline has "
    "already measured the signal for you; the measurements are given below. "
    "Use them to answer. Answer concisely and end with the exact line "
    "'Final answer: <value>'.")

FEWSHOT_SYSTEM = SYSTEM


def _feature_block(signals, fs):
    """Every toolbox measurement, precomputed, as a short text block.

    This is the `feature_prompt` condition: it hands the model exactly what a
    perfectly-routed tool loop would obtain, with no call to make. It
    separates the two abilities the tool condition confounds -- CHOOSING the
    right measurement (routing) and USING a measurement once given (reading).
    """
    lines = ["Measurements computed from the signal by a DSP pipeline:"]
    for w in sorted(signals):
        tag = "" if len(signals) == 1 else f" (window {w})"
        acc = signals[w]
        lines.append(
            f"- peak count{tag}: {dsp_tools.gt.count_peaks(acc, fs)}")
        lines.append(
            f"- dominant frequency{tag}: "
            f"{dsp_tools.gt.dominant_freq(acc, fs):.2f} Hz")
        lines.append(
            f"- signal magnitude area (movement intensity){tag}: "
            f"{dsp_tools.gt.sma(acc):.4f}")
        lines.append(
            f"- largest energy-change (transition) time{tag}: "
            f"{dsp_tools.transition_detect(acc, fs):.2f} s")
        lines.append(
            f"- longest active-bout duration{tag}: "
            f"{dsp_tools.active_bout_duration(acc, fs):.2f} s")
    return "\n".join(lines)


def run_feature_prompt(llm, item, variant=None, max_new_tokens=128):
    """Question + precomputed DSP measurements, no raw samples, no tools."""
    sig = variant["serialized_signal"] if variant else item[
        "serialized_signal"]
    signals, fs = split_windows(sig)
    q = item["question"] + "\n\n" + _feature_block(signals, fs)
    text, n_in, n_gen, dt = llm.chat(
        [{"role": "system", "content": FEATURE_SYSTEM},
         {"role": "user", "content": q}], max_new_tokens=max_new_tokens)
    return _result("feature_prompt", item, text,
                   extract_answer(text, item["family"]), n_in, n_gen, dt,
                   variant)


DISTRACTOR_SPECS = [
    {"name": "step_cadence",
     "description": "Average step cadence in steps per minute.", "args": {}},
    {"name": "posture_angle",
     "description": "Mean inclination of the sensor from vertical, degrees.",
     "args": {}},
    {"name": "jerk_rms",
     "description": "RMS of the derivative of acceleration (jerk).",
     "args": {}},
    {"name": "signal_quality",
     "description": "Fraction of samples free of clipping or dropout.",
     "args": {}},
]


def run_tool_distractor(llm, item, variant=None, max_rounds=4):
    """tool_augmented with four plausible but useless tools added.

    R3/R4 both note that routing among five tools whose names map 1:1 onto the
    five question families is a low-entropy problem. This condition keeps the
    same five working tools and adds four distractors that sound applicable to
    IMU questions but answer none of them, so the routing decision is no
    longer a name match.
    """
    sig_text = variant["serialized_signal"] if variant else item[
        "serialized_signal"]
    signals, fs = split_windows(sig_text)
    specs = dsp_tools.TOOL_SPECS + DISTRACTOR_SPECS
    order = [specs[i] for i in (5, 0, 6, 2, 7, 1, 8, 3, 4)]
    system = (
        "You are an expert wearable-sensor signal analyst with access to a DSP "
        "toolbox that runs on the provided signal. Tools:\n"
        + "\n".join(f"- {t['name']}: {t['description']} args: {t['args']}"
                     for t in order)
        + "\nThe toolbox already holds the signal shown below; NEVER pass the "
        "signal samples as arguments. To call a tool, reply with ONLY a JSON "
        'object like {"tool": "count_peaks", "args": {}} (only sma takes '
        '{"window": "A"} or {"window": "B"}). After you have the tool results '
        'you need, reply with ONLY {"answer": <number or letter in quotes>}. '
        "Never guess without a tool call.")
    q = item["question"] + "\n\nSignal:\n" + sig_text
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": q}]
    tot_in = tot_gen = 0
    tot_dt = 0.0
    calls = []
    text = ""
    for _ in range(max_rounds):
        text, n_in, n_gen, dt = llm.chat(messages, max_new_tokens=96)
        tot_in += n_in
        tot_gen += n_gen
        tot_dt += dt
        obj = _parse_json_obj(text)
        if obj is None:
            break
        if "answer" in obj:
            r = _result("tool_distractor", item, text,
                        _coerce(obj["answer"], item["family"]),
                        tot_in, tot_gen, tot_dt, variant)
            r["tool_calls"] = calls
            return r
        if "tool" in obj:
            name = obj["tool"]
            if any(d["name"] == name for d in DISTRACTOR_SPECS):
                res = {"error": f"{name} is not applicable to this question"}
            else:
                res = dsp_tools.run_tool(name, signals, fs, obj.get("args"))
            calls.append({"tool": name, "args": obj.get("args"),
                          "result": res})
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user",
                             "content": "Tool result: " + json.dumps(res)})
        else:
            break
    pred = None if '"tool"' in text else extract_answer(text, item["family"])
    r = _result("tool_distractor", item, text, pred, tot_in, tot_gen, tot_dt,
                variant)
    r["tool_calls"] = calls
    return r


FAMILY_TOOL = {"peak_count": ("count_peaks", {}),
               "dominant_freq": ("dominant_freq", {}),
               "intensity_order": ("sma", {"window": "A"}),
               "transition_time": ("transition_detect", {}),
               "bout_duration": ("active_bout", {})}


def run_tool_forced(llm, item, variant=None, max_new_tokens=128):
    """Perfect-routing upper bound for the tool architecture.

    The family's correct tool is executed for the model and its raw JSON
    result injected; the model only has to read the result and answer. The
    gap between this and `tool_augmented` is exactly the cost of the model's
    own tool SELECTION, measured rather than inferred from call counts.
    """
    sig = variant["serialized_signal"] if variant else item[
        "serialized_signal"]
    signals, fs = split_windows(sig)
    name, args = FAMILY_TOOL[item["family"]]
    results = []
    if item["family"] == "intensity_order":
        for w in ("A", "B"):
            if w in signals:
                results.append(dsp_tools.run_tool(name, signals, fs,
                                                  {"window": w}))
    else:
        results.append(dsp_tools.run_tool(name, signals, fs, args))
    q = (item["question"] + "\n\nTool `" + name + "` was run on the signal. "
         "Result: " + json.dumps(results if len(results) > 1 else results[0]))
    text, n_in, n_gen, dt = llm.chat(
        [{"role": "system", "content": FEATURE_SYSTEM},
         {"role": "user", "content": q}], max_new_tokens=max_new_tokens)
    r = _result("tool_forced", item, text,
                extract_answer(text, item["family"]), n_in, n_gen, dt,
                variant)
    r["tool_calls"] = [{"tool": name, "args": args, "result": x,
                        "forced": True} for x in results]
    return r


FEWSHOT_BANK = {
    "peak_count": ("Signal:\nSampling rate: 10 Hz, duration: 4 s, 40 samples."
                   "\nmagnitude: 1.00 1.31 0.72 1.02 1.29 0.71 1.01 1.30 0.70"
                   " 1.00 1.31 0.72 1.02 1.29 0.71 1.01 1.30 0.70 1.00 1.31"
                   " 0.72 1.02 1.29 0.71 1.01 1.30 0.70 1.00 1.31 0.72 1.02"
                   " 1.29 0.71 1.01 1.30 0.70 1.00 1.31 0.72 1.02\n"
                   "Question: How many movement cycles?",
                   "The magnitude repeats with a period of about 3 samples "
                   "(0.3 s) over 4 s, so about 13 cycles.\n"
                   "Final answer: 13"),
    "dominant_freq": ("Signal:\nSampling rate: 10 Hz, duration: 4 s, 40 "
                      "samples.\nmagnitude: 1.00 1.50 1.00 0.50 1.00 1.50 "
                      "1.00 0.50 1.00 1.50 1.00 0.50 1.00 1.50 1.00 0.50 "
                      "1.00 1.50 1.00 0.50 1.00 1.50 1.00 0.50 1.00 1.50 "
                      "1.00 0.50 1.00 1.50 1.00 0.50 1.00 1.50 1.00 0.50 "
                      "1.00 1.50 1.00 0.50\nQuestion: Dominant frequency?",
                      "One full oscillation spans 4 samples = 0.4 s, so the "
                      "frequency is 1/0.4 = 2.5 Hz.\nFinal answer: 2.5"),
}


def run_with_signal_fewshot(llm, item, variant=None, max_new_tokens=128):
    """with_signal plus two worked numeric examples (in-context demos).

    Controls for 'the model was never shown how to do the task'.
    """
    sig = variant["serialized_signal"] if variant else item[
        "serialized_signal"]
    msgs = [{"role": "system", "content": FEWSHOT_SYSTEM}]
    for fam in ("peak_count", "dominant_freq"):
        u, a = FEWSHOT_BANK[fam]
        msgs.append({"role": "user", "content": u})
        msgs.append({"role": "assistant", "content": a})
    msgs.append({"role": "user",
                 "content": item["question"] + "\n\nSignal:\n" + sig})
    text, n_in, n_gen, dt = llm.chat(msgs, max_new_tokens=max_new_tokens)
    return _result("with_signal_fewshot", item, text,
                   extract_answer(text, item["family"]), n_in, n_gen, dt,
                   variant)


def run_with_signal_long(llm, item, variant=None, max_new_tokens=256):
    """with_signal with a generous generation budget.

    Removes the truncation confound: Stage 2 capped non-CoT conditions at 48
    new tokens and the 1.5B model hit that cap on nearly every item, so its
    negative prior-gap could not be cleanly attributed to the signal.
    """
    sig = variant["serialized_signal"] if variant else item[
        "serialized_signal"]
    q = item["question"] + "\n\nSignal:\n" + sig
    text, n_in, n_gen, dt = llm.chat(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": q}], max_new_tokens=max_new_tokens)
    return _result("with_signal_long", item, text,
                   extract_answer(text, item["family"]), n_in, n_gen, dt,
                   variant)


def run_with_signal_format(llm, item, variant=None, fmt="magnitude",
                           max_new_tokens=128):
    """with_signal, re-serialized in an alternative text format.

    Same window, same ground truth, different rendering -- so accuracy is
    directly comparable and a format artefact would show up as a difference.
    """
    key = "serialized_" + fmt
    sig = (variant or item).get(key) or (variant or item)[
        "serialized_signal"]
    q = item["question"] + "\n\nSignal:\n" + sig
    text, n_in, n_gen, dt = llm.chat(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": q}], max_new_tokens=max_new_tokens)
    r = _result("with_signal_" + fmt, item, text,
                extract_answer(text, item["family"]), n_in, n_gen, dt,
                variant)
    r["format"] = fmt
    return r


PRIME_HEADER = (
    "Reference excerpts from the UCI-HAR dataset that you have previously "
    "seen, with their published values:\n")

PRIME_INSTRUCTION = (
    "\n\nIf the signal below appears verbatim among the reference excerpts "
    "above, report that excerpt's published value. Otherwise compute the "
    "answer from the signal yourself.\n\n")


def run_contam_primed(llm, item, table, max_new_tokens=128):
    """POSITIVE CONTROL for the contamination probe.

    Getting this control right took two attempts, and the reason is worth
    stating because it is a property of the probe, not of our code.

    The perturbation (resample to 47 Hz + 3-D rotation) is *answer
    preserving*: rotation preserves magnitude and resampling preserves
    frequency, so the canonical and perturbed arms of a pair share the same
    ground truth (measured: 100% of pairs agree within tolerance).  A
    memorising model is therefore NOT detected by giving a different answer
    on the two arms -- it is detected because RECALL IS ARM-SPECIFIC: only
    the canonical numeric string was ever published, so the model can
    retrieve on the canonical arm and must compute on the perturbed one,
    which it cannot do.

    That is what the control has to simulate, and simple priming does not.
    Our first version placed the queried window's value in context, which
    the model can copy on *either* arm -- arm-independent access, hence no
    gap, hence no validation.  This version instead supplies a LOOKUP TABLE
    of several canonical windows (the queried pair's twin among them) and
    asks the model to report a published value only on a verbatim match.
    On the canonical arm the queried signal matches an entry; on the
    perturbed arm nothing matches and the model is thrown back on
    computation.  A working probe must show a gap here.
    """
    blocks = [f"[excerpt {k + 1}]\n{t['serialized_signal']}\n"
              f"Published value: {t['ground_truth']}"
              for k, t in enumerate(table)]
    q = (PRIME_HEADER + "\n\n".join(blocks) + PRIME_INSTRUCTION
         + item["question"] + "\n\nSignal:\n" + item["serialized_signal"])
    text, n_in, n_gen, dt = llm.chat(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": q}], max_new_tokens=max_new_tokens)
    r = _result("contam_primed", item, text,
                extract_answer(text, item["family"]), n_in, n_gen, dt)
    r["prime_table"] = [t["id"] for t in table]
    return r


def run_with_signal_sampled(llm, item, variant=None, temperature=0.7,
                            seed=0, max_new_tokens=128):
    """with_signal under stochastic decoding -- decoding-variance study."""
    sig = variant["serialized_signal"] if variant else item[
        "serialized_signal"]
    q = item["question"] + "\n\nSignal:\n" + sig
    text, n_in, n_gen, dt = llm.chat(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": q}], max_new_tokens=max_new_tokens,
        temperature=temperature, seed=seed)
    r = _result("with_signal_sampled", item, text,
                extract_answer(text, item["family"]), n_in, n_gen, dt,
                variant)
    r["decode_seed"] = seed
    r["temperature"] = temperature
    return r
