"""GroundCheck benchmark generator.

Builds QA items with programmatic ground truth, counterfactual variants, and
contamination probes. Deterministic under --seed. Writes JSONL + config.

Item schema:
  id, dataset, family, question, serialized_signal, ground_truth,
  tolerance {type, value[, min_abs]}, answer_format, meta {...},
  counterfactual_variants [ {type, serialized_signal, ground_truth,
                             expected_direction, meta} ]
"""
import numpy as np

import ground_truth as gt
from serialize import serialize

FMT_NUM = "Respond with only a single number."
FMT_AB = "Respond with only the single letter A or B."

DEVICE_PHRASE = {"ucihar": "a smartphone worn at the waist",
                 "wisdm": "a smartphone carried in the pocket",
                 "pamap2": "a chest-mounted IMU"}


def _q_peak(w, dur):
    act = w["activity"].replace("_", " ")
    return (f"The following is a {dur:.0f}-second tri-axial accelerometer "
            f"recording ({w['meta']['units']}) from {DEVICE_PHRASE[w['dataset']]} "
            f"while the subject was {act}. How many repetitive movement-cycle "
            f"peaks (e.g., step impacts) occur in the acceleration magnitude "
            f"over the whole recording? {FMT_NUM}")


def _q_freq(w, dur):
    act = w["activity"].replace("_", " ")
    return (f"The following is a {dur:.0f}-second tri-axial accelerometer "
            f"recording ({w['meta']['units']}) from {DEVICE_PHRASE[w['dataset']]} "
            f"while the subject was {act}. What is the dominant frequency, in "
            f"Hz, of the acceleration magnitude? {FMT_NUM}")


def _q_intensity(wa, wb, dur):
    aa = wa["activity"].replace("_", " ")
    ab = wb["activity"].replace("_", " ")
    return (f"Below are two {dur:.0f}-second tri-axial accelerometer windows "
            f"from {DEVICE_PHRASE[wa['dataset']]}. Window A was recorded while "
            f"the subject was {aa}; window B while {ab}. Which window shows "
            f"higher overall movement intensity? {FMT_AB}")


def _q_transition(w, dur, frm, to):
    return (f"The following is a {dur:.0f}-second tri-axial accelerometer "
            f"recording ({w['meta']['units']}) from {DEVICE_PHRASE[w['dataset']]}. "
            f"The subject changes activity from {frm.replace('_', ' ')} to "
            f"{to.replace('_', ' ')} exactly once. At what time, in seconds "
            f"from the start, does the transition occur? {FMT_NUM}")


def _q_bout(w, dur, act):
    return (f"The following is a {dur:.0f}-second tri-axial accelerometer "
            f"recording ({w['meta']['units']}) from {DEVICE_PHRASE[w['dataset']]}. "
            f"It contains a single contiguous bout of {act.replace('_', ' ')} "
            f"surrounded by rest. How long, in seconds, does the active bout "
            f"last? {FMT_NUM}")


def _meta(w, ser_fs, extra=None):
    m = {"subject": w["subject"], "activity": w["activity"],
         "source_fs": w["fs"], "serialized_fs": ser_fs,
         "units": w["meta"]["units"]}
    if extra:
        m.update(extra)
    return m


class Generator:
    def __init__(self, seed=0, target_fs=10.0, decimals=2, max_samples=120):
        self.rng = np.random.default_rng(seed)
        self.tfs = target_fs
        self.dec = decimals
        self.maxs = max_samples
        self.items = []
        self._n = 0

    def _nid(self):
        self._n += 1
        return f"gc-{self._n:05d}"

    def _ser(self, acc, fs, max_samples=None):
        return serialize(acc, fs, self.tfs, self.dec,
                         max_samples or self.maxs)

    # ---------------------------------------------------------- families --

    def add_peak_count(self, w, with_cf=True):
        acc, fs = w["acc"], w["fs"]
        text, ds, ds_fs = self._ser(acc, fs)
        n_ds = ds.shape[0]
        n_full = int(round(n_ds * fs / ds_fs))
        acc = acc[:n_full]
        dur = n_ds / ds_fs
        gt_val = gt.count_peaks(acc, fs)
        if gt_val < 3:
            return None
        item = {"id": self._nid(), "dataset": w["dataset"],
                "family": "peak_count", "question": _q_peak(w, dur),
                "serialized_signal": text, "ground_truth": float(gt_val),
                "tolerance": {"type": "relative", "value": 0.2, "min_abs": 2},
                "answer_format": FMT_NUM, "meta": _meta(w, ds_fs),
                "counterfactual_variants": []}
        if with_cf:
            self._add_signal_cfs(item, w, acc, fs, "peak_count")
        self.items.append(item)
        return item

    def add_dominant_freq(self, w, with_cf=True):
        acc, fs = w["acc"], w["fs"]
        text, ds, ds_fs = self._ser(acc, fs)
        n_full = int(round(ds.shape[0] * fs / ds_fs))
        acc = acc[:n_full]
        dur = ds.shape[0] / ds_fs
        f0 = gt.dominant_freq(acc, fs, max_freq=min(5.0, 0.45 * self.tfs))
        if f0 <= 0.3:
            return None
        item = {"id": self._nid(), "dataset": w["dataset"],
                "family": "dominant_freq", "question": _q_freq(w, dur),
                "serialized_signal": text, "ground_truth": round(float(f0), 3),
                "tolerance": {"type": "absolute", "value": 0.3},
                "answer_format": FMT_NUM, "meta": _meta(w, ds_fs),
                "counterfactual_variants": []}
        if with_cf:
            self._add_signal_cfs(item, w, acc, fs, "dominant_freq")
        self.items.append(item)
        return item

    def _oracle(self, family, acc, fs):
        if family == "peak_count":
            return float(gt.count_peaks(acc, fs))
        if family == "dominant_freq":
            return round(float(gt.dominant_freq(
                acc, fs, max_freq=min(5.0, 0.45 * self.tfs))), 3)
        raise ValueError(family)

    def _add_signal_cfs(self, item, w, acc, fs, family):
        """signal_swap, scramble, time_stretch counterfactuals for
        peak_count / dominant_freq items."""
        cfs = item["counterfactual_variants"]
        # scramble
        seed = int(self.rng.integers(0, 2 ** 31))
        scr = gt.phase_scramble(acc, seed=seed)
        t_scr, _, _ = self._ser(scr, fs)
        cfs.append({"type": "scramble", "serialized_signal": t_scr,
                    "ground_truth": self._oracle(family, scr, fs),
                    "expected_direction": None,
                    "meta": {"seed": seed}})
        # time stretch (slower cadence)
        factor = float(self.rng.choice([1.25, 1.5]))
        st = gt.time_stretch(acc, factor)
        t_st, _, _ = self._ser(st, fs)
        cfs.append({"type": "time_stretch", "serialized_signal": t_st,
                    "ground_truth": self._oracle(family, st, fs),
                    "expected_direction": "decrease",
                    "meta": {"factor": factor}})

    def add_swap_cf(self, item, w_swap):
        """signal_swap: signal from a DIFFERENT activity; question unchanged."""
        acc, fs = w_swap["acc"], w_swap["fs"]
        text, ds, ds_fs = self._ser(acc, fs)
        n_full = int(round(ds.shape[0] * fs / ds_fs))
        gtv = self._oracle(item["family"], acc[:n_full], fs)
        item["counterfactual_variants"].append(
            {"type": "signal_swap", "serialized_signal": text,
             "ground_truth": gtv, "expected_direction": None,
             "meta": {"swap_activity": w_swap["activity"],
                      "swap_subject": w_swap["subject"]}})

    def add_intensity(self, wa, wb, with_cf=True):
        ta, dsa, ds_fs = self._ser(wa["acc"], wa["fs"])
        tb, dsb, _ = self._ser(wb["acc"], wb["fs"])
        dur = dsa.shape[0] / ds_fs
        na = int(round(dsa.shape[0] * wa["fs"] / ds_fs))
        nb = int(round(dsb.shape[0] * wb["fs"] / ds_fs))
        order = gt.intensity_order(wa["acc"][:na], wb["acc"][:nb])
        sma_a, sma_b = gt.sma(wa["acc"][:na]), gt.sma(wb["acc"][:nb])
        # require unambiguous ordering (>1.5x ratio)
        if max(sma_a, sma_b) < 1.5 * min(sma_a, sma_b):
            return None
        text = f"Window A:\n{ta}\nWindow B:\n{tb}"
        item = {"id": self._nid(), "dataset": wa["dataset"],
                "family": "intensity_order",
                "question": _q_intensity(wa, wb, dur),
                "serialized_signal": text, "ground_truth": order,
                "tolerance": {"type": "exact"}, "answer_format": FMT_AB,
                "meta": _meta(wa, ds_fs,
                              {"activity_b": wb["activity"],
                               "subject_b": wb["subject"],
                               "sma_a": round(sma_a, 4),
                               "sma_b": round(sma_b, 4)}),
                "counterfactual_variants": []}
        if with_cf:
            # pair_swap: present the same two windows in the reversed roles
            text_rev = f"Window A:\n{tb}\nWindow B:\n{ta}"
            item["counterfactual_variants"].append(
                {"type": "pair_swap", "serialized_signal": text_rev,
                 "ground_truth": "B" if order == "A" else "A",
                 "expected_direction": "flip", "meta": {}})
            # amplitude_edit: damp the currently-stronger window by 4x -> flip
            strong, weak = (wa, wb) if order == "A" else (wb, wa)
            scale = 0.25
            damped = strong["acc"] * scale + strong["acc"].mean(axis=0) * (1 - scale)
            td, _, _ = self._ser(damped, strong["fs"])
            if order == "A":
                text_amp = f"Window A:\n{td}\nWindow B:\n{tb}"
                new_gt = gt.intensity_order(damped[:na], wb["acc"][:nb])
            else:
                text_amp = f"Window A:\n{ta}\nWindow B:\n{td}"
                new_gt = gt.intensity_order(wa["acc"][:na], damped[:nb])
            item["counterfactual_variants"].append(
                {"type": "amplitude_edit", "serialized_signal": text_amp,
                 "ground_truth": new_gt,
                 "expected_direction": ("flip" if new_gt != order else None),
                 "meta": {"scale": scale}})
        self.items.append(item)
        return item

    def add_transition(self, w, with_cf=True, tol=2.0):
        """Real transition window (PAMAP2) with per-sample labels.

        Ground truth comes from the leave-one-placement-out consensus of the
        two simultaneous IMUs the model is NOT shown (see
        loaders.pamap2_consensus_transitions); the +-2 s tolerance is 10% of
        the 20 s window and reflects the measured cross-placement spread of a
        real whole-body transition.
        """
        acc, fs = w["acc"], w["fs"]
        # 20 s transition windows need the full span serialized (200 samples
        # at 10 Hz), not the 120-sample cap used for 12 s activity windows.
        text, ds, ds_fs = self._ser(acc, fs, max_samples=max(
            self.maxs, int(round(acc.shape[0] / fs * self.tfs))))
        dur = ds.shape[0] / ds_fs
        t_true = gt.transition_time(w["labels"], fs)
        if t_true is None or t_true < 2.0 or t_true > dur - 2.0:
            return None
        frm, to = w["meta"]["from"], w["meta"]["to"]
        extra = {"from": frm, "to": to, "real_transition": True}
        for k in ("placement", "gt_source", "held_out_placements",
                  "consensus_s", "label_boundary_s", "annotation_lag_s"):
            if k in w["meta"]:
                extra[k] = w["meta"][k]
        item = {"id": self._nid(), "dataset": w["dataset"],
                "family": "transition_time",
                "question": _q_transition(w, dur, frm, to),
                "serialized_signal": text,
                "ground_truth": round(float(t_true), 2),
                "tolerance": {"type": "absolute", "value": tol},
                "answer_format": FMT_NUM,
                "meta": _meta(w, ds_fs, extra),
                "counterfactual_variants": []}
        if with_cf:
            self._transition_cfs(item, acc, fs, t_true, dur)
        self.items.append(item)
        return item

    def add_spliced_transition(self, wa, wb, frac, with_cf=True):
        """Constructed transition: first frac of activity A + tail of B."""
        fs = wa["fs"]
        n = min(wa["acc"].shape[0], wb["acc"].shape[0])
        sp = gt.splice(wa["acc"][:n], wb["acc"][:n], frac)
        text, ds, ds_fs = self._ser(sp, fs)
        dur = ds.shape[0] / ds_fs
        n_full = int(round(ds.shape[0] * fs / ds_fs))
        t_true = frac * (n / fs)
        if t_true < 2.0 or t_true > dur - 2.0:
            return None
        w = dict(wa)
        w["meta"] = dict(wa["meta"])
        item = {"id": self._nid(), "dataset": wa["dataset"],
                "family": "transition_time",
                "question": _q_transition(w, dur, wa["activity"],
                                          wb["activity"]),
                "serialized_signal": text,
                "ground_truth": round(float(t_true), 2),
                "tolerance": {"type": "absolute", "value": 1.0},
                "answer_format": FMT_NUM,
                "meta": _meta(w, ds_fs, {"from": wa["activity"],
                                         "to": wb["activity"],
                                         "subject_b": wb["subject"],
                                         "real_transition": False,
                                         "splice_frac": frac}),
                "counterfactual_variants": []}
        if with_cf:
            self._transition_cfs(item, sp[:n_full], fs, t_true, dur)
        self.items.append(item)
        return item

    def _transition_cfs(self, item, acc, fs, t_true, dur):
        cfs = item["counterfactual_variants"]
        n = acc.shape[0]
        k = int(round(t_true * fs))
        # directional shift: move the transition later by cropping the head
        # of segment A? Instead re-splice: repeat A's tail to delay B.
        shift = min(2.5, (dur - 2.0) - t_true)
        if shift >= 1.5:
            m = int(round(shift * fs))
            a_ext = np.concatenate([acc[:k], acc[max(0, k - m):k],
                                    acc[k:]], axis=0)[:n]
            t_sh, _, _ = self._ser(a_ext, fs)
            cfs.append({"type": "transition_shift",
                        "serialized_signal": t_sh,
                        "ground_truth": round(float(t_true + shift), 2),
                        "expected_direction": "increase",
                        "meta": {"shift_s": shift}})
        # scramble: transition destroyed; a grounded model must NOT still
        # answer the original time (scored as hallucination probe)
        seed = int(self.rng.integers(0, 2 ** 31))
        scr = gt.phase_scramble(acc, seed=seed)
        t_scr, _, _ = self._ser(scr, fs)
        cfs.append({"type": "scramble", "serialized_signal": t_scr,
                    "ground_truth": None, "expected_direction": None,
                    "meta": {"seed": seed,
                             "note": "no true transition; grounded answer "
                                     "should not equal original time"}})

    def add_bout(self, w_rest, w_act, bout_frac, with_cf=True):
        """Constructed rest|active|rest window from two real segments."""
        fs = w_rest["fs"]
        n = min(w_rest["acc"].shape[0], w_act["acc"].shape[0])
        # compose on the length that survives serialization (max_samples cap)
        n = min(n, int(round(self.maxs / self.tfs * fs)))
        rest, act = w_rest["acc"][:n], w_act["acc"][:n]
        nb = int(round(n * bout_frac))
        s0 = (n - nb) // 2
        comp = np.concatenate([rest[:s0], act[:nb], rest[s0 + nb:]], axis=0)
        text, ds, ds_fs = self._ser(comp, fs)
        dur = ds.shape[0] / ds_fs
        n_full = int(round(ds.shape[0] * fs / ds_fs))
        bout_s = nb / fs
        if bout_s < 2.0 or bout_s > dur - 2.0:
            return None
        item = {"id": self._nid(), "dataset": w_rest["dataset"],
                "family": "bout_duration",
                "question": _q_bout(w_rest, dur, w_act["activity"]),
                "serialized_signal": text,
                "ground_truth": round(float(bout_s), 2),
                "tolerance": {"type": "relative", "value": 0.25,
                              "min_abs": 1.0},
                "answer_format": FMT_NUM,
                "meta": _meta(w_rest, ds_fs,
                              {"rest_activity": w_rest["activity"],
                               "active_activity": w_act["activity"],
                               "bout_frac": bout_frac,
                               "constructed": True}),
                "counterfactual_variants": []}
        if with_cf and bout_frac <= 0.5:
            # directional edit: longer bout (increase)
            nb2 = int(round(n * min(0.8, bout_frac + 0.25)))
            s02 = (n - nb2) // 2
            comp2 = np.concatenate([rest[:s02], act[:nb2],
                                    rest[s02 + nb2:]], axis=0)
            t2, _, _ = self._ser(comp2, fs)
            item["counterfactual_variants"].append(
                {"type": "bout_stretch", "serialized_signal": t2,
                 "ground_truth": round(float(nb2 / fs), 2),
                 "expected_direction": "increase",
                 "meta": {"bout_frac": round(nb2 / n, 3)}})
            # scramble
            seed = int(self.rng.integers(0, 2 ** 31))
            scr = gt.phase_scramble(comp[:n_full], seed=seed)
            t3, _, _ = self._ser(scr, fs)
            item["counterfactual_variants"].append(
                {"type": "scramble", "serialized_signal": t3,
                 "ground_truth": None, "expected_direction": None,
                 "meta": {"seed": seed}})
        self.items.append(item)
        return item

    # ------------------------------------------------- contamination -----

    def add_contamination_pair(self, w, family):
        """Canonical UCI-HAR test window serialized at native 50 Hz (verbatim
        values, 3 decimals) vs the SAME window resampled to 47 Hz + randomly
        rotated. Both arms mention the dataset name to elicit memorization."""
        acc, fs = w["acc"], w["fs"]
        act = w["activity"].replace("_", " ")
        dur = acc.shape[0] / fs
        qbase = (f"The following is a {dur:.2f}-second tri-axial accelerometer "
                 f"recording (units: g, 50 Hz) from the UCI-HAR smartphone "
                 f"dataset, recorded while the subject was {act}. ")
        if family == "peak_count":
            q = qbase + ("How many repetitive movement-cycle peaks (e.g., "
                         "step impacts) occur in the acceleration magnitude? "
                         + FMT_NUM)
            tol = {"type": "relative", "value": 0.25, "min_abs": 2}
        else:
            q = qbase + ("What is the dominant frequency, in Hz, of the "
                         "acceleration magnitude? " + FMT_NUM)
            tol = {"type": "absolute", "value": 0.4}
        out = []
        pair_id = f"pair-{self._n + 1:05d}"
        for arm in ("canonical", "perturbed"):
            if arm == "canonical":
                a2, fs2 = acc, fs
            else:
                seed = int(self.rng.integers(0, 2 ** 31))
                a2 = gt.rotate(gt.resample_rate(acc, fs, 47.0), seed=seed)
                fs2 = 47.0
            gtv = (float(gt.count_peaks(a2, fs2)) if family == "peak_count"
                   else round(float(gt.dominant_freq(a2, fs2, max_freq=5.0)),
                              3))
            if family == "peak_count" and gtv < 2:
                return []
            if family == "dominant_freq" and gtv <= 0.3:
                return []
            fmt = "%.3f"
            lines = [f"Sampling rate: {fs2:g} Hz, {a2.shape[0]} samples "
                     f"per axis."]
            for j, name in enumerate("xyz"):
                lines.append(f"{name}: " + " ".join(fmt % v
                                                    for v in a2[:, j]))
            item = {"id": self._nid(), "dataset": "ucihar",
                    "family": family, "question": q,
                    "serialized_signal": "\n".join(lines),
                    "ground_truth": gtv, "tolerance": tol,
                    "answer_format": FMT_NUM,
                    "meta": _meta(w, fs2, {"contamination_probe": arm,
                                           "pair_id": pair_id,
                                           "native_serialization": True}),
                    "counterfactual_variants": []}
            self.items.append(item)
            out.append(item)
        return out
