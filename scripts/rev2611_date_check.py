"""gpt-oss date-line check (26 Nov 2026): resend raw-text items whose answer
changed between the 25 Nov and 26 Nov runs with the prompt date pinned to
each day, through the verified raw-prompt path (src/ollama_llm.py).
Result: results/rev2611/date_check.json."""
import json, os, sys, time, requests
os.chdir(os.path.expanduser("~/Documents/SonHa/Percom27/groundcheck-sensor-llm-grounding-audit"))
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
import evalplan, harness
import ollama_llm as O
items = {i["id"]: i for i in evalplan.load_items("data/benchmark/items.jsonl")}
old = {}
for l in open("results/raw_rev2611/grid_gpt-oss-120b-thinklow.jsonl"):
    r = json.loads(l)
    if r["backend"] == "with_signal":
        old[r["item_id"]] = r
new = {}
for l in open("results/rev2611/version_check_rows.jsonl"):
    r = json.loads(l)
    if r["tag"] == "gpt-oss-120b-thinklow":
        new[r["item_id"]] = r
changed = [i for i in new if new[i]["old_pred"] != new[i]["new_pred"]][:4]
U = "http://127.0.0.1:11441/api/generate"
opts = {"num_predict": 1024 + 2048, "num_ctx": 8192, "seed": 0, "temperature": 0.0, "top_k": 1}
real = time.strftime
for iid in changed:
    it = items[iid]
    msgs = [{"role": "system", "content": harness.SYSTEM},
            {"role": "user", "content": it["question"] + "\n\nSignal:\n" + it["serialized_signal"]}]
    out = {}
    for d in ("2026-11-25", "2026-11-26"):
        O.time.strftime = (lambda fmt, *a, _d=d: _d if fmt == "%Y-%m-%d" else real(fmt, *a))
        g = requests.post(U, json={"model": "gpt-oss:120b", "raw": True, "stream": False, "options": opts,
                                   "prompt": O.render_harmony(msgs, "low")}, timeout=3600).json()
        content, _ = O.parse_harmony(g.get("response", ""))
        out[d] = harness.extract_answer(content, it["family"])
    O.time.strftime = real
    print(iid, it["family"], "recorded 25 Nov:", old[iid]["pred"], "| raw 25 Nov:", out["2026-11-25"],
          "| recorded 26 Nov:", new[iid]["new_pred"], "| raw 26 Nov:", out["2026-11-26"], flush=True)
