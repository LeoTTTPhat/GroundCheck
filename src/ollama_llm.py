"""Ollama chat backend with the same interface as harness.HFLLM.

Added in the November 2026 revision so that 2025-2026 open-weight models
(Qwen3, Mistral Small 3.1, Gemma 4, gpt-oss) run the unchanged GroundCheck
conditions. Every backend in harness.py calls ``llm.chat(messages,
max_new_tokens=..., temperature=..., seed=...)`` and expects
``(text, prompt_tokens, gen_tokens, latency_s)``; this class honours that.

Reasoning models. Ollama counts reasoning ("thinking") tokens against
``num_predict``. A model allowed to reason therefore gets an extra
``think_budget`` on top of each backend's answer cap, so the answer budget is
the same as for the original six models. Per-query reasoning statistics are
accumulated in ``self.meta`` and attached to the record by the runner.

Native tool-call syntax (v2, 2026-11-25). Ollama's per-model output parsers
remove a model's native tool-call tokens from ``message.content``; when the
request declares no tools (our protocol asks for a plain JSON object in the
reply instead) the call is dropped and the reply arrives empty. Gemma 4, for
example, answers the chosen-tool prompt with
``<|tool_call>call:count_peaks{}<tool_call|>``. We therefore request the
generated tokens (``logprobs``), rebuild the reply from them when the parsed
content is empty, and rewrite a native call into the protocol's form
``{"tool": NAME, "args": {...}}``. Prompts are unchanged.
"""
import json
import re
import os
import time

import requests

# A dedicated server (GROUNDCHECK_OLLAMA=http://127.0.0.1:PORT) keeps these
# runs from sharing a scheduler with other jobs on the default port.
URL = os.environ.get("GROUNDCHECK_OLLAMA", "http://127.0.0.1:11434") + "/api/chat"


class OllamaLLM:
    def __init__(self, model, think=None, think_budget=0, num_ctx=8192,
                 timeout=900):
        self.name = model
        self.think = think            # None (model default), False, True,
        self.think_budget = int(think_budget)   # or "low"/"medium"/"high"
        self.num_ctx = num_ctx
        self.timeout = timeout
        # gpt-oss's template puts the current date in the system prompt, and
        # the date alone changes answers. GROUNDCHECK_HARMONY_DATE pins it:
        # every gpt-oss turn then goes through the raw path with that date,
        # which reproduces Ollama's own output for the same date exactly.
        self.harmony_date = (os.environ.get("GROUNDCHECK_HARMONY_DATE")
                             if model.startswith("gpt-oss") else None)
        self.device = "ollama-metal"
        self.dtype = "ollama-default-quant"
        self.supports_system = True
        self.stable = {"ok": True, "note": "ollama backend"}
        self.reset_meta()

    def reset_meta(self):
        self.meta = {"calls": 0, "thinking_chars": 0, "thinking_tokens_est": 0,
                     "done_reasons": [], "empty_content": 0}

    def verify_on(self, probe_items):
        return self.stable

    def _harmony_raw(self, messages, opts):
        """Fallback for gpt-oss when Ollama's harmony parser fails (HTTP
        500, e.g. on a call to the model's built-in python tool): send the
        identically rendered prompt with raw=True and parse it ourselves."""
        level = self.think if isinstance(self.think, str) else "medium"
        r = requests.post(URL.replace("/api/chat", "/api/generate"),
                          json={"model": self.name, "raw": True,
                                "stream": False, "options": opts,
                                "keep_alive": "30m",
                                "prompt": render_harmony(
                                    messages, level, self.harmony_date)},
                          timeout=self.timeout)
        r.raise_for_status()
        g = r.json()
        content, thinking = parse_harmony(g.get("response", ""))
        key = "harmony_pinned" if self.harmony_date else "harmony_fallback"
        self.meta[key] = self.meta.get(key, 0) + 1
        if self.harmony_date:
            self.meta["harmony_date"] = self.harmony_date
        return {"message": {"content": content, "thinking": thinking},
                "prompt_eval_count": g.get("prompt_eval_count", 0),
                "eval_count": g.get("eval_count", 0),
                "done_reason": g.get("done_reason")}

    def chat(self, messages, max_new_tokens=128, temperature=None, seed=None):
        opts = {"num_predict": int(max_new_tokens) + self.think_budget,
                "num_ctx": self.num_ctx, "seed": 0 if seed is None else seed}
        if temperature is None:
            opts.update(temperature=0.0, top_k=1)     # greedy, as in the audit
        else:
            opts.update(temperature=float(temperature), top_p=0.95, top_k=50)
        body = {"model": self.name, "messages": messages, "stream": False,
                "options": opts, "keep_alive": "30m", "logprobs": True}
        if self.think is not None:
            body["think"] = self.think
        t0 = time.time()
        for attempt in range(3):
            try:
                if self.harmony_date:
                    js = self._harmony_raw(messages, opts)
                    break
                r = requests.post(URL, json=body, timeout=self.timeout)
                if r.status_code == 500 and self.name.startswith("gpt-oss"):
                    js = self._harmony_raw(messages, opts)
                    break
                r.raise_for_status()
                js = r.json()
                break
            except requests.RequestException:
                if attempt == 2:
                    raise
                time.sleep(5)
        dt = time.time() - t0
        msg = js.get("message", {})
        text = msg.get("content", "") or ""
        thinking = msg.get("thinking", "") or ""
        if not text.strip() and not thinking.strip() and js.get("logprobs"):
            raw = "".join(t.get("token", "") for t in js["logprobs"])
            if raw.strip():
                text = raw
                self.meta["recovered_from_tokens"] = \
                    self.meta.get("recovered_from_tokens", 0) + 1
        conv = native_to_protocol(text)
        if conv is not None:
            if "unparsed_native_call" in conv:
                self.meta.setdefault("unparsed_native", []).append(text[:300])
            text = conv
            self.meta["native_tool_call"] = \
                self.meta.get("native_tool_call", 0) + 1
        self.meta["calls"] += 1
        self.meta["thinking_chars"] += len(thinking)
        self.meta["thinking_tokens_est"] += len(thinking) // 4
        self.meta["done_reasons"].append(js.get("done_reason"))
        if not text.strip():
            self.meta["empty_content"] += 1
        return (text, int(js.get("prompt_eval_count", 0)),
                int(js.get("eval_count", 0)), dt)


def render_harmony(messages, level, date=None):
    """gpt-oss prompt exactly as Ollama's template renders it for a request
    without tools (system message -> developer '# Instructions'; earlier
    assistant turns as 'final' content; thinking of earlier turns dropped)."""
    sys_msg = next((m["content"] for m in messages
                    if m["role"] == "system"), None)
    s = ("<|start|>system<|message|>You are ChatGPT, a large language model "
         "trained by OpenAI.\nKnowledge cutoff: 2024-06\nCurrent date: "
         + (date or time.strftime("%Y-%m-%d")) + "\n\nReasoning: " + str(level) +
         "\n\n# Valid channels: analysis, commentary, final. Channel must "
         "be included for every message.<|end|>")
    if sys_msg:
        s += ("<|start|>developer<|message|>\n\n# Instructions\n\n"
              + sys_msg + "<|end|>")
    for m in messages:
        if m["role"] == "user":
            s += "<|start|>user<|message|>" + m["content"] + "<|end|>"
        elif m["role"] == "assistant" and m.get("content"):
            s += ("<|start|>assistant<|channel|>final<|message|>"
                  + m["content"] + "<|end|>")
    return s + "<|start|>assistant"


def parse_harmony(raw):
    """Split raw harmony output into (content, thinking) the way Ollama's
    parser does: analysis-channel text is thinking; commentary and final
    messages are joined in order as content. A call to the built-in python
    tool (a header with ``to=python``) becomes a fenced code block, the
    code-agent protocol's own form."""
    content, thinking = [], []
    for seg in re.split(r"<\|start\|>", raw):
        if "<\|message\|>" not in seg and "<|message|>" not in seg:
            continue
        head, body = seg.split("<|message|>", 1)
        body = re.split(r"<\|end\|>|<\|call\|>|<\|return\|>", body)[0]
        if "to=" in head:
            if "python" in head:
                content.append("```python\n" + body.strip() + "\n```")
            continue
        if "<|channel|>analysis" in head:
            thinking.append(body)
        else:
            content.append(body)
    return "".join(content), " ".join(thinking)


def _loose_json(s):
    s = s.replace('<|"|>', '"').strip()
    if not s:
        return {}
    for cand in (s, s.replace("'", '"')):
        try:
            return json.loads(cand)
        except ValueError:
            pass
        c2 = re.sub(r'([{,]\s*)([A-Za-z_]\w*)\s*:', r'\1"\2":', cand)
        c2 = re.sub(r':\s*([A-Za-z_]\w*)\s*([,}])', r': "\1"\2', c2)
        try:
            return json.loads(c2)
        except ValueError:
            pass
    return {}


_NAME = r"(?:[A-Za-z_][\w.]*:)?([A-Za-z_]\w*)"   # optional namespace prefix


def native_to_protocol(text):
    """Rewrite a native tool call (Gemma 4, Mistral) as the protocol's JSON
    tool request; return None when the text holds no native call. A native
    call that cannot be parsed becomes a request for an unknown tool (the
    toolbox answers with an error), so that no number is ever scraped out of
    a tool call's arguments."""
    m = re.search(r"<\|tool_call>\s*(?:call:)?\s*(\{.*\})\s*<tool_call\|>",
                  text, re.S)
    if m:                              # the protocol's own JSON, wrapped
        obj = _loose_json(m.group(1))
        if isinstance(obj, dict) and ("tool" in obj or "answer" in obj):
            return json.dumps(obj)
    m = re.search(r"<\|tool_call>\s*call:\s*" + _NAME +
                  r"\s*(\{.*?\})?\s*<tool_call\|>", text, re.S)
    if m:
        return json.dumps({"tool": m.group(1),
                           "args": _loose_json(m.group(2) or "{}")})
    m = re.search(r"\[TOOL_CALLS\]\s*" + _NAME + r"\s*\[ARGS\]\s*(\{.*\})?",
                  text, re.S)
    if m:
        return json.dumps({"tool": m.group(1),
                           "args": _loose_json(m.group(2) or "{}")})
    m = re.search(r"\[TOOL_CALLS\]\s*(\[.*\])", text, re.S)
    if m:
        try:
            c = json.loads(m.group(1))[0]
            return json.dumps({"tool": str(c.get("name")).split(":")[-1],
                               "args": c.get("arguments") or {}})
        except (ValueError, IndexError, AttributeError, KeyError):
            pass
    if "<|tool_call>" in text or "[TOOL_CALLS]" in text:
        return json.dumps({"tool": "unparsed_native_call", "args": {}})
    return None


if __name__ == "__main__":
    for t in ['<|tool_call>call:count_peaks{}<tool_call|><|tool_response>',
              '<|tool_call>call:sma{window:<|"|>B<|"|>}<tool_call|>',
              "<|tool_call>call:dsp_toolbox:sma{window: 'A'}<tool_call|>",
              '<|tool_call>call:transition_detect<tool_call|>',
              '<|tool_call>garbled<tool_call|>',
              '<|tool_call>call:{"tool": "sma", "args": {"window": "B"}}<tool_call|>',
              '[TOOL_CALLS]active_bout[ARGS]{}',
              '[TOOL_CALLS][{"name": "sma", "arguments": {"window": "A"}}]',
              '{"answer": 3}', 'Final answer: 12']:
        print(repr(t), "->", native_to_protocol(t))
