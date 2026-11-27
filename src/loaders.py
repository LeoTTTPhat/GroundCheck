"""Dataset loaders for GroundCheck.

Every loader yields Window dicts with a COMMON schema:
  {dataset, subject, activity (str), fs (Hz), acc (N,3 float, m/s^2 or g as
   noted), labels (optional per-sample activity array for transition/bout
   windows), meta {...}}

UCI-HAR : raw inertial signals (total_acc), 128-sample 2.56 s windows @50 Hz,
          units of g. Windows overlap 50%; loader can also stitch consecutive
          windows of one (subject, experiment-contiguous) block into longer
          continuous segments by taking the first half of each window.
WISDM   : UCI #507 raw phone/watch accel, 20 Hz, m/s^2, per-subject activity
          files (A=walking, B=jogging, ...).
PAMAP2  : Protocol .dat files @100 Hz, chest IMU 16g accel (m/s^2), continuous
          per-sample activity labels -> real transitions and bouts.
"""
import os
import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
RAW = os.path.join(DATA, "raw")

UCIHAR_ACT = {1: "walking", 2: "walking_upstairs", 3: "walking_downstairs",
              4: "sitting", 5: "standing", 6: "laying"}

WISDM_ACT = {"A": "walking", "B": "jogging", "C": "stairs", "D": "sitting",
             "E": "standing", "F": "typing", "G": "brushing_teeth",
             "H": "eating_soup", "I": "eating_chips", "J": "eating_pasta",
             "K": "drinking", "L": "eating_sandwich", "M": "kicking_soccer",
             "O": "playing_catch", "P": "dribbling_basketball", "Q": "writing",
             "R": "clapping", "S": "folding_clothes"}

PAMAP2_ACT = {1: "lying", 2: "sitting", 3: "standing", 4: "walking",
              5: "running", 6: "cycling", 7: "nordic_walking",
              12: "ascending_stairs", 13: "descending_stairs",
              16: "vacuum_cleaning", 17: "ironing", 24: "rope_jumping"}


# ------------------------------------------------------------- UCI-HAR ----

def _ucihar_dir():
    for cand in (os.path.join(RAW, "UCI HAR Dataset"),
                 os.path.join(RAW, "ucihar", "UCI HAR Dataset")):
        if os.path.isdir(cand):
            return cand
    raise FileNotFoundError("UCI HAR Dataset not found under data/raw")


def load_ucihar(split="train"):
    """Return list of Window dicts, one per 128-sample raw window (units: g)."""
    d = _ucihar_dir()
    sig = {}
    for ax in "xyz":
        p = os.path.join(d, split, "Inertial Signals",
                         f"total_acc_{ax}_{split}.txt")
        sig[ax] = np.loadtxt(p)
    y = np.loadtxt(os.path.join(d, split, f"y_{split}.txt")).astype(int)
    subj = np.loadtxt(os.path.join(d, split,
                                   f"subject_{split}.txt")).astype(int)
    out = []
    for i in range(len(y)):
        acc = np.stack([sig["x"][i], sig["y"][i], sig["z"][i]], axis=1)
        out.append({"dataset": "ucihar", "subject": int(subj[i]),
                    "activity": UCIHAR_ACT[int(y[i])], "fs": 50.0,
                    "acc": acc, "labels": None,
                    "meta": {"split": split, "window_index": i,
                             "units": "g"}})
    return out


def stitch_ucihar(windows, min_windows=4):
    """Stitch runs of consecutive same-subject same-activity 50%-overlap
    windows into longer continuous segments (first 64 samples of each window
    + full last window)."""
    segs = []
    cur = []
    prev = None
    for w in windows:
        key = (w["subject"], w["activity"], w["meta"]["split"])
        idx = w["meta"]["window_index"]
        if prev is not None and key == prev[0] and idx == prev[1] + 1:
            cur.append(w)
        else:
            if len(cur) >= min_windows:
                segs.append(_stitch(cur))
            cur = [w]
        prev = (key, idx)
    if len(cur) >= min_windows:
        segs.append(_stitch(cur))
    return segs


def _stitch(run):
    parts = [w["acc"][:64] for w in run[:-1]] + [run[-1]["acc"]]
    acc = np.concatenate(parts, axis=0)
    w0 = run[0]
    return {"dataset": "ucihar", "subject": w0["subject"],
            "activity": w0["activity"], "fs": 50.0, "acc": acc,
            "labels": None,
            "meta": {"split": w0["meta"]["split"],
                     "window_index": w0["meta"]["window_index"],
                     "n_windows": len(run), "units": "g", "stitched": True}}


# --------------------------------------------------------------- WISDM ----

def _wisdm_dir():
    for cand in (os.path.join(RAW, "wisdm-dataset"),
                 os.path.join(RAW, "wisdm", "wisdm-dataset")):
        if os.path.isdir(cand):
            return cand
    raise FileNotFoundError("wisdm-dataset not found under data/raw")


def load_wisdm(device="phone", subjects=None, activities=("A", "B", "C"),
               window_s=10.0, max_windows_per_file=6):
    """Non-overlapping windows from WISDM raw accel (units: m/s^2, 20 Hz)."""
    d = os.path.join(_wisdm_dir(), "raw", device, "accel")
    fs = 20.0
    out = []
    files = sorted(os.listdir(d))
    for fn in files:
        if not fn.endswith(".txt"):
            continue
        sid = int(fn.split("_")[1])
        if subjects is not None and sid not in subjects:
            continue
        per_act = {}
        with open(os.path.join(d, fn)) as f:
            for line in f:
                line = line.strip().rstrip(";")
                if not line:
                    continue
                p = line.split(",")
                if len(p) < 6:
                    continue
                act = p[1]
                if act not in activities:
                    continue
                try:
                    per_act.setdefault(act, []).append(
                        (float(p[3]), float(p[4]), float(p[5])))
                except ValueError:
                    continue
        n = int(window_s * fs)
        for act, rows in per_act.items():
            arr = np.array(rows)
            k = 0
            for s in range(0, len(arr) - n + 1, n):
                if k >= max_windows_per_file:
                    break
                out.append({"dataset": "wisdm", "subject": sid,
                            "activity": WISDM_ACT.get(act, act), "fs": fs,
                            "acc": arr[s:s + n], "labels": None,
                            "meta": {"device": device, "start": s,
                                     "units": "m/s^2"}})
                k += 1
    return out


WISDM11_ACT = {"Walking": "walking", "Jogging": "jogging",
               "Upstairs": "walking_upstairs",
               "Downstairs": "walking_downstairs",
               "Sitting": "sitting", "Standing": "standing"}


def load_wisdm_v11(activities=("Walking", "Jogging", "Upstairs",
                               "Downstairs", "Sitting", "Standing"),
                   window_s=12.0, max_windows_per_key=4):
    """Fordham WISDM v1.1 raw file (20 Hz phone accel, m/s^2).

    Lines: user,activity,timestamp_ns,x,y,z; Groups rows by (user, activity)
    into contiguous chunks (timestamp gap < 1 s) and tiles non-overlapping
    windows.
    """
    path = os.path.join(RAW, "WISDM_ar_v1.1", "WISDM_ar_v1.1_raw.txt")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    fs = 20.0
    data = {}
    with open(path) as f:
        for line in f:
            line = line.strip().rstrip(";")
            if not line:
                continue
            p = line.split(",")
            if len(p) < 6:
                continue
            try:
                uid, act, ts = int(p[0]), p[1], int(p[2])
                xyz = (float(p[3]), float(p[4]), float(p[5]))
            except ValueError:
                continue
            if act not in activities:
                continue
            data.setdefault((uid, act), []).append((ts, xyz))
    out = []
    n = int(window_s * fs)
    for (uid, act), rows in sorted(data.items()):
        rows.sort(key=lambda r: r[0])
        # split on gaps > 1 s (ns timestamps); timestamp 0 rows -> own chunk
        chunks = []
        cur = []
        prev = None
        for ts, xyz in rows:
            if prev is not None and (ts <= 0 or ts - prev > 1e9):
                if len(cur) >= n:
                    chunks.append(cur)
                cur = []
            cur.append(xyz)
            prev = ts
        if len(cur) >= n:
            chunks.append(cur)
        k = 0
        for ch in chunks:
            arr = np.array(ch)
            for s in range(0, len(arr) - n + 1, n):
                if k >= max_windows_per_key:
                    break
                out.append({"dataset": "wisdm", "subject": uid,
                            "activity": WISDM11_ACT[act], "fs": fs,
                            "acc": arr[s:s + n], "labels": None,
                            "meta": {"device": "phone_v11", "start": s,
                                     "units": "m/s^2"}})
                k += 1
    return out


# -------------------------------------------------------------- PAMAP2 ----

def _pamap2_dir():
    for cand in (os.path.join(RAW, "PAMAP2_Dataset"),
                 os.path.join(RAW, "pamap2", "PAMAP2_Dataset")):
        if os.path.isdir(cand):
            return cand
    raise FileNotFoundError("PAMAP2_Dataset not found under data/raw")


def load_pamap2_subject(subject_file):
    """Load one Protocol .dat: (t, activity_id, chest 16g acc (N,3) m/s^2).

    Column map (0-based): 0 time, 1 activityID, 2 HR, 3..19 hand IMU,
    20..36 chest IMU, 37..53 ankle IMU. Within an IMU block: 0 temp,
    1..3 acc16g, 4..6 acc6g, 7..9 gyro, 10..12 mag, 13..16 orientation.
    Chest acc16g = columns 21..23.
    """
    import pandas as pd
    raw = pd.read_csv(subject_file, sep=" ", header=None,
                      usecols=[0, 1, 21, 22, 23],
                      dtype=float, na_values=["NaN"]).values
    t = raw[:, 0]
    act = raw[:, 1].astype(int)
    acc = raw[:, 2:5]
    good = ~np.isnan(acc).any(axis=1)
    # forward-fill short NaN gaps instead of dropping (keeps time grid)
    if not good.all():
        idx = np.where(good, np.arange(len(acc)), 0)
        np.maximum.accumulate(idx, out=idx)
        acc = acc[idx]
    return t, act, acc


def pamap2_activity_windows(t, act, acc, fs=100.0, window_s=10.0,
                            activities=None, max_per_activity=8, subject=0):
    """Clean single-activity windows (no label 0 transient)."""
    n = int(window_s * fs)
    out = []
    counts = {}
    # find contiguous runs of a single non-zero activity
    change = np.nonzero(np.diff(act) != 0)[0] + 1
    bounds = np.concatenate([[0], change, [len(act)]])
    for s, e in zip(bounds[:-1], bounds[1:]):
        a = act[s]
        if a == 0 or a not in PAMAP2_ACT:
            continue
        if activities and PAMAP2_ACT[a] not in activities:
            continue
        # skip first/last 5 s of the run (settling), then tile windows
        s2, e2 = s + int(5 * fs), e - int(5 * fs)
        for ws in range(s2, e2 - n + 1, n * 3):
            if counts.get(a, 0) >= max_per_activity:
                break
            out.append({"dataset": "pamap2", "subject": subject,
                        "activity": PAMAP2_ACT[a], "fs": fs,
                        "acc": acc[ws:ws + n], "labels": None,
                        "meta": {"start_idx": int(ws), "units": "m/s^2"}})
            counts[a] = counts.get(a, 0) + 1
    return out


def pamap2_transition_windows(t, act, acc, fs=100.0, window_s=20.0,
                              max_windows=20, subject=0, seed=0):
    """Windows containing exactly one real activity transition (A -> B), with
    per-sample labels. Label 0 (transient) gaps are bridged only when short
    (<3 s); the transition time is the midpoint of the gap."""
    rng = np.random.default_rng(seed)
    n = int(window_s * fs)
    change = np.nonzero(np.diff(act) != 0)[0] + 1
    bounds = np.concatenate([[0], change, [len(act)]])
    runs = [(s, e, act[s]) for s, e in zip(bounds[:-1], bounds[1:])]
    out = []
    for i in range(len(runs) - 2):
        s1, e1, a1 = runs[i]
        s2, e2, a2 = runs[i + 1]
        # pattern A, 0-gap, B  (or direct A, B)
        if a1 == 0 or a1 not in PAMAP2_ACT:
            continue
        if a2 == 0:
            if i + 2 >= len(runs):
                continue
            s3, e3, a3 = runs[i + 2]
            if a3 == 0 or a3 not in PAMAP2_ACT or a3 == a1:
                continue
            gap = (e2 - s2) / fs
            if gap > 3.0:
                continue
            t_mid, next_act, next_start = (s2 + e2) // 2, a3, s3
        else:
            if a2 not in PAMAP2_ACT or a2 == a1:
                continue
            t_mid, next_act, next_start = s2, a2, s2
        # place transition at a randomized offset within the window
        off = int(rng.uniform(0.25, 0.75) * n)
        ws = t_mid - off
        we = ws + n
        if ws < s1 or we > (next_start + (e2 if a2 != 0 else e3) - next_start):
            pass
        if ws < 0 or we > len(act):
            continue
        # require window covered by run A then run B (plus short gap)
        labels = np.where(np.arange(ws, we) < t_mid, a1, next_act)
        # verify actual labels match (outside the short gap)
        actual = act[ws:we]
        ok = (actual == labels) | (actual == 0)
        if ok.mean() < 0.97:
            continue
        out.append({"dataset": "pamap2", "subject": subject,
                    "activity": f"{PAMAP2_ACT[a1]}->{PAMAP2_ACT[next_act]}",
                    "fs": fs, "acc": acc[ws:we], "labels": labels,
                    "meta": {"start_idx": int(ws), "units": "m/s^2",
                             "from": PAMAP2_ACT[a1],
                             "to": PAMAP2_ACT[next_act],
                             "transition_s": float((t_mid - ws) / fs)}})
        if len(out) >= max_windows:
            break
    return out


# --- PAMAP2 multi-placement + exact label-derived transition events --------
# (Stage 3) The Protocol files carry three IMUs (hand / chest / ankle) and
# long label-0 transients between activities.  Rather than guessing a
# midpoint inside a transient we take the EXACT label-derived event: the last
# sample still labelled A, i.e. the instant the subject stops activity A.
# That is unambiguous however long the following transient is, and it is the
# same quantity the energy-change detector localises.

PAMAP2_IMU_COLS = {"hand": (4, 5, 6), "chest": (21, 22, 23),
                   "ankle": (38, 39, 40)}


def load_pamap2_subject_multi(subject_file, placements=("hand", "chest",
                                                        "ankle")):
    """Load one Protocol/Optional .dat -> (t, activity_id, {placement: (N,3)}).

    Column map (0-based): 0 time, 1 activityID, 2 HR, 3..19 hand IMU,
    20..36 chest IMU, 37..53 ankle IMU; within a block 0 temp, 1..3 acc16g.
    """
    import pandas as pd
    cols = [0, 1]
    for pl in placements:
        cols.extend(PAMAP2_IMU_COLS[pl])
    raw = pd.read_csv(subject_file, sep=" ", header=None,
                      usecols=sorted(cols), dtype=float,
                      na_values=["NaN"]).values
    order = {c: k for k, c in enumerate(sorted(cols))}
    t = raw[:, order[0]]
    act = raw[:, order[1]].astype(int)
    out = {}
    for pl in placements:
        a = raw[:, [order[c] for c in PAMAP2_IMU_COLS[pl]]]
        good = ~np.isnan(a).any(axis=1)
        if not good.all():
            idx = np.where(good, np.arange(len(a)), 0)
            np.maximum.accumulate(idx, out=idx)
            a = a[idx]
        out[pl] = a
    return t, act, out


def pamap2_stop_events(act, fs=100.0, min_run_s=12.0, max_gap_s=25.0):
    """Every 'subject stops activity A' event with an exact label boundary.

    Returns [(idx_last_A, activity_A, activity_next, run_start)] where
    idx_last_A is the index of the final sample carrying label A.
    """
    change = np.nonzero(np.diff(act) != 0)[0] + 1
    bounds = np.concatenate([[0], change, [len(act)]])
    runs = [(s, e, int(act[s])) for s, e in zip(bounds[:-1], bounds[1:])]
    ev = []
    for i, (s, e, a) in enumerate(runs):
        if a == 0 or a not in PAMAP2_ACT:
            continue
        if (e - s) / fs < min_run_s:
            continue
        # find the next real activity after A (possibly across one transient)
        nxt = None
        j = i + 1
        gap = 0.0
        while j < len(runs):
            s2, e2, a2 = runs[j]
            if a2 == 0:
                gap += (e2 - s2) / fs
                if gap > max_gap_s:
                    break
                j += 1
                continue
            if a2 in PAMAP2_ACT and a2 != a:
                nxt = a2
            break
        if nxt is None:
            continue
        ev.append((e - 1, a, nxt, s))
    return ev


def pamap2_real_transition_windows(subject_file, subject, fs=100.0,
                                   window_s=20.0, placements=("hand",
                                                              "chest",
                                                              "ankle"),
                                   offsets=(0.4, 0.65), seed=0):
    """Real PAMAP2 transition windows: exact label-derived stop-of-A time.

    One window per (stop event x placement x offset). `labels` is the
    per-sample activity id (A before the event, `next` after), so the
    generic label-derived ground truth in ground_truth.transition_time
    reproduces the event exactly.
    """
    t, act, accs = load_pamap2_subject_multi(subject_file, placements)
    n = int(window_s * fs)
    out = []
    for (idx_last, a, nxt, run_start) in pamap2_stop_events(act, fs=fs):
        for off in offsets:
            ws = int(idx_last + 1 - off * n)
            we = ws + n
            if ws < run_start or we > len(act):
                continue
            labels = np.where(np.arange(ws, we) <= idx_last, a, nxt)
            for pl in placements:
                out.append({
                    "dataset": "pamap2", "subject": subject,
                    "activity": f"{PAMAP2_ACT[a]}->{PAMAP2_ACT[nxt]}",
                    "fs": fs, "acc": accs[pl][ws:we], "labels": labels,
                    "meta": {"start_idx": int(ws), "units": "m/s^2",
                             "placement": pl, "from": PAMAP2_ACT[a],
                             "to": PAMAP2_ACT[nxt], "real_transition": True,
                             "transition_s": float((idx_last + 1 - ws) / fs)}})
    return out


def pamap2_clean_windows(subject_file, subject, fs=100.0, window_s=12.0,
                         placements=("hand", "chest", "ankle"),
                         max_per_activity=3, activities=None):
    """Clean single-activity PAMAP2 windows across all three IMU placements."""
    t, act, accs = load_pamap2_subject_multi(subject_file, placements)
    n = int(window_s * fs)
    change = np.nonzero(np.diff(act) != 0)[0] + 1
    bounds = np.concatenate([[0], change, [len(act)]])
    out, counts = [], {}
    for s, e in zip(bounds[:-1], bounds[1:]):
        a = int(act[s])
        if a == 0 or a not in PAMAP2_ACT:
            continue
        if activities and PAMAP2_ACT[a] not in activities:
            continue
        s2, e2 = s + int(5 * fs), e - int(5 * fs)
        for ws in range(s2, e2 - n + 1, n * 2):
            if counts.get(a, 0) >= max_per_activity:
                break
            for pl in placements:
                out.append({"dataset": "pamap2", "subject": subject,
                            "activity": PAMAP2_ACT[a], "fs": fs,
                            "acc": accs[pl][ws:ws + n], "labels": None,
                            "meta": {"start_idx": int(ws),
                                     "units": "m/s^2", "placement": pl}})
            counts[a] = counts.get(a, 0) + 1
    return out


def pamap2_files():
    """All Protocol + Optional .dat paths, sorted."""
    import glob
    d = _pamap2_dir()
    return (sorted(glob.glob(os.path.join(d, "Protocol", "*.dat"))) +
            sorted(glob.glob(os.path.join(d, "Optional", "*.dat"))))


def pamap2_consensus_transitions(subject_file, subject, fs=100.0,
                                 window_s=20.0,
                                 placements=("hand", "chest", "ankle"),
                                 offsets=(0.35, 0.5, 0.65),
                                 agree_s=1.5, agree_all_s=1.5,
                                 search_s=12.0, seed=0):
    """Real PAMAP2 transitions with LEAVE-ONE-PLACEMENT-OUT ground truth.

    PAMAP2's activity-end annotations lag the accelerometer energy boundary
    (measured median 4.7 s, Sec. `annotation lag` in the paper), so the raw
    label boundary is not a usable +-1.5 s target.  Instead each item's
    ground truth is the consensus changepoint of the TWO IMUs the model is
    NOT shown, kept only when those two independently agree within
    `agree_s` and the consensus lies within `search_s` of the annotated
    boundary.  The ground truth therefore comes from sensors absent from the
    prompt, so it is not circular with respect to the item's own signal.

    Returns windows with `labels` rewritten so the generic label-derived
    ground truth reproduces the consensus instant exactly.
    """
    import dsp_tools as _dt
    t, act, accs = load_pamap2_subject_multi(subject_file, placements)
    n = int(window_s * fs)
    out = []
    for (idx_last, a, nxt, run_start) in pamap2_stop_events(act, fs=fs):
        for off in offsets:
            ws = int(idx_last + 1 - off * n)
            we = ws + n
            if ws < run_start or we > len(act):
                continue
            label_t = (idx_last + 1 - ws) / fs
            cps = {pl: _dt.transition_detect(accs[pl][ws:we], fs)
                   for pl in placements}
            vals = list(cps.values())
            # item-validity gate: keep only whole-body transitions that ALL
            # three simultaneous IMUs localise to the same instant. This is
            # a property of the recording, not of any model.
            if max(vals) - min(vals) > agree_all_s:
                continue
            for pl in placements:
                others = [cps[q] for q in placements if q != pl]
                if len(others) < 2:
                    continue
                if abs(others[0] - others[1]) > agree_s:
                    continue
                cons = float(np.mean(others))
                if abs(cons - label_t) > search_s:
                    continue
                if cons < 2.0 or cons > window_s - 2.0:
                    continue
                k = int(round(cons * fs))
                labels = np.where(np.arange(n) < k, a, nxt)
                out.append({
                    "dataset": "pamap2", "subject": subject,
                    "activity": f"{PAMAP2_ACT[a]}->{PAMAP2_ACT[nxt]}",
                    "fs": fs, "acc": accs[pl][ws:we], "labels": labels,
                    "meta": {"start_idx": int(ws), "units": "m/s^2",
                             "placement": pl, "from": PAMAP2_ACT[a],
                             "to": PAMAP2_ACT[nxt], "real_transition": True,
                             "gt_source": "loo_consensus",
                             "held_out_placements": [q for q in placements
                                                     if q != pl],
                             "consensus_s": round(cons, 2),
                             "label_boundary_s": round(label_t, 2),
                             "annotation_lag_s": round(label_t - cons, 2)}})
    return out
