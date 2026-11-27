"""DSP toolbox exposed to the tool-augmented LLM backend and used verbatim by
the dsp_oracle upper-bound backend.

Tools operate on the SERIALIZED (downsampled, rounded) signal that the model
receives in its prompt -- i.e., exactly the information available to any
prompted model -- so the dsp_oracle is an honest ceiling on prompt-extractable
performance, including serialization loss.
"""
import numpy as np

import ground_truth as gt

TOOL_SPECS = [
    {"name": "count_peaks",
     "description": "Count movement-cycle peaks (e.g., steps/repetitions) in "
                    "the acceleration magnitude of the provided signal.",
     "args": {}},
    {"name": "dominant_freq",
     "description": "Dominant frequency in Hz of the acceleration magnitude "
                    "(Welch PSD).",
     "args": {}},
    {"name": "sma",
     "description": "Signal magnitude area (movement intensity). Pass "
                    "window='A' or window='B' when two windows are given.",
     "args": {"window": "A|B (optional, default A)"}},
    {"name": "transition_detect",
     "description": "Detect the time in seconds of the largest change in "
                    "short-term signal energy (activity transition).",
     "args": {}},
    {"name": "active_bout",
     "description": "Duration in seconds of the longest contiguous "
                    "high-movement (active) bout in the signal.",
     "args": {}},
]


def active_bout_duration(acc, fs, smooth_s=1.5):
    """Duration (s) of the longest contiguous run where the smoothed movement
    energy envelope exceeds a 2-means (Lloyd/Otsu-style) threshold."""
    acc = np.asarray(acc, dtype=float)
    hp = acc - acc.mean(axis=0, keepdims=True)
    e = np.abs(hp).sum(axis=1)
    w = max(1, int(smooth_s * fs))
    kern = np.ones(w) / w
    env = np.convolve(e, kern, mode="same")
    thr = 0.5 * (np.percentile(env, 10) + np.percentile(env, 90))
    for _ in range(20):
        lo, hi = env[env <= thr], env[env > thr]
        if len(lo) == 0 or len(hi) == 0:
            break
        new = 0.5 * (lo.mean() + hi.mean())
        if abs(new - thr) < 1e-9:
            break
        thr = new
    mask = (env > thr).astype(int)
    if mask.sum() == 0:
        return 0.0
    edges = np.diff(np.concatenate([[0], mask, [0]]))
    starts = np.nonzero(edges == 1)[0]
    ends = np.nonzero(edges == -1)[0]
    return float((ends - starts).max() / fs)


def transition_detect(acc, fs, smooth_s=0.5, edge_s=2.0):
    """Exact single-changepoint detector on the movement-energy envelope.

    Maximises the between-segment sum of squares of the smoothed
    |high-passed acceleration| envelope over all split points (the exact
    least-squares mean-shift changepoint, computed in O(n) from cumulative
    sums).  Replaces the fixed 2 s sliding-window difference used in Stage 2:
    on real PAMAP2 transitions it cuts the disagreement between simultaneous
    body-worn IMUs from a median 2.49 s to 1.84 s, and it raises the spliced
    UCI-HAR ceiling from 0.933 to 0.950.
    """
    acc = np.asarray(acc, dtype=float)
    hp = acc - acc.mean(axis=0, keepdims=True)
    e = np.abs(hp).sum(axis=1)
    w = max(1, int(smooth_s * fs))
    if w > 1:
        e = np.convolve(e, np.ones(w) / w, mode="same")
    n = len(e)
    edge = max(1, int(edge_s * fs))
    if n < 2 * edge + 2:
        return float(n / (2 * fs))
    c = np.concatenate([[0.0], np.cumsum(e)])
    tot = c[-1]
    k = np.arange(edge, n - edge)
    gain = c[k] ** 2 / k + (tot - c[k]) ** 2 / (n - k)
    return float(k[int(np.argmax(gain))] / fs)


def run_tool(name, item_signals, fs, args=None):
    """Execute tool `name` on the item's serialized signal(s).

    item_signals: dict with 'A' (N,3) and optionally 'B'.
    Returns a JSON-serializable result dict.
    """
    args = args or {}
    which = str(args.get("window", "A")).strip().upper()[:1]
    acc = item_signals.get(which if which in item_signals else "A")
    if name == "count_peaks":
        return {"peaks": gt.count_peaks(acc, fs)}
    if name == "dominant_freq":
        return {"dominant_freq_hz": round(gt.dominant_freq(acc, fs), 3)}
    if name == "sma":
        return {"window": which, "sma": round(gt.sma(acc), 4)}
    if name == "transition_detect":
        return {"transition_s": round(transition_detect(acc, fs), 2)}
    if name == "active_bout":
        return {"active_bout_s": round(active_bout_duration(acc, fs), 2)}
    return {"error": f"unknown tool {name}"}


def oracle_answer(family, item_signals, fs):
    """dsp_oracle backend: deterministic answer from the serialized signal."""
    if family == "peak_count":
        return float(gt.count_peaks(item_signals["A"], fs))
    if family == "dominant_freq":
        return float(round(gt.dominant_freq(item_signals["A"], fs), 3))
    if family == "intensity_order":
        return gt.intensity_order(item_signals["A"], item_signals["B"])
    if family == "transition_time":
        return float(round(transition_detect(item_signals["A"], fs), 2))
    if family == "bout_duration":
        return float(round(active_bout_duration(item_signals["A"], fs), 2))
    raise ValueError(family)
