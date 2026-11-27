"""Unit tests validating the ground-truth oracle on synthetic signals with
KNOWN answers. Run: python3 -m pytest src/test_ground_truth.py -q
(or python3 src/test_ground_truth.py to run without pytest)."""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ground_truth as gt  # noqa: E402


def make_sine_acc(freq, fs, dur, amp=2.0, gravity=9.81, seed=0):
    """Triaxial accel mimicking vertical gait: gravity on z with the sinusoid
    superimposed ALONG the gravity axis (as vertical body acceleration is), so
    |acc| oscillates at `freq` rather than at the rectified 2*freq."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * fs)) / fs
    acc = np.zeros((len(t), 3))
    acc[:, 2] = gravity + amp * np.sin(2 * np.pi * freq * t)
    acc += rng.normal(0, 0.01, acc.shape)
    return acc


def test_peak_count_sine():
    fs, dur = 50.0, 10.0
    for freq in (1.0, 2.0, 3.0):
        acc = make_sine_acc(freq, fs, dur)
        n = gt.count_peaks(acc, fs)
        expected = freq * dur
        assert abs(n - expected) <= 1, (freq, n, expected)


def test_peak_count_pulse_train():
    """Gaussian pulse train: exactly 7 pulses at known times."""
    fs, dur = 50.0, 10.0
    t = np.arange(int(dur * fs)) / fs
    times = np.linspace(1.0, 9.0, 7)
    sig = np.zeros_like(t)
    for c in times:
        sig += 3.0 * np.exp(-((t - c) ** 2) / (2 * 0.12 ** 2))
    acc = np.zeros((len(t), 3))
    acc[:, 2] = 9.81 + sig
    n = gt.count_peaks(acc, fs)
    assert abs(n - 7) <= 1, n


def test_dominant_freq():
    fs, dur = 50.0, 12.0
    for freq in (0.8, 1.6, 2.5, 4.0):
        acc = make_sine_acc(freq, fs, dur)
        f = gt.dominant_freq(acc, fs)
        assert abs(f - freq) <= 0.26, (freq, f)  # Welch bin width ~0.25 Hz


def test_intensity_order():
    fs = 50.0
    lo = make_sine_acc(2.0, fs, 8.0, amp=0.5, seed=1)
    hi = make_sine_acc(2.0, fs, 8.0, amp=4.0, seed=2)
    assert gt.intensity_order(hi, lo) == "A"
    assert gt.intensity_order(lo, hi) == "B"
    assert gt.sma(hi) > gt.sma(lo)


def test_transition_time():
    fs = 50.0
    labels = np.array([0] * 150 + [1] * 350)  # change at sample 150 -> 3.0 s
    assert abs(gt.transition_time(labels, fs) - 3.0) < 1e-9
    assert gt.transition_time(np.zeros(100), fs) is None


def test_all_transitions():
    fs = 10.0
    labels = np.array([0] * 20 + [1] * 30 + [0] * 10)
    assert gt.all_transition_times(labels, fs) == [2.0, 5.0]


def test_bout_duration():
    fs = 50.0
    labels = np.array([1] * 100 + [2] * 250 + [1] * 400 + [2] * 50)
    assert abs(gt.bout_duration(labels, fs, 1) - 8.0) < 1e-9   # 400/50
    assert abs(gt.bout_duration(labels, fs, 2) - 5.0) < 1e-9   # 250/50
    assert gt.bout_duration(labels, fs, 9) == 0.0


def test_time_stretch_changes_freq():
    fs = 50.0
    acc = make_sine_acc(2.0, fs, 10.0)
    slower = gt.time_stretch(acc, 1.25)   # 2.0 -> 1.6 Hz
    faster = gt.time_stretch(acc, 0.8)    # 2.0 -> 2.5 Hz
    assert abs(gt.dominant_freq(slower, fs) - 1.6) <= 0.3
    assert abs(gt.dominant_freq(faster, fs) - 2.5) <= 0.3


def test_phase_scramble_preserves_spectrum_destroys_phase():
    fs = 50.0
    acc = make_sine_acc(2.0, fs, 10.0)
    scr = gt.phase_scramble(acc, seed=3)
    # dominant frequency survives (magnitude spectrum kept per axis)
    assert abs(gt.dominant_freq(scr[:, 2], fs) - 2.0) <= 0.3
    # variance preserved per axis
    assert np.allclose(scr.var(axis=0), acc.var(axis=0), rtol=0.05, atol=1e-6)
    # temporal structure destroyed: an impulsive pulse train becomes
    # Gaussian-like noise (excess kurtosis collapses toward 0)
    t = np.arange(int(10 * fs)) / fs
    sig = np.zeros_like(t)
    for c in np.linspace(1.0, 9.0, 5):
        sig += 5.0 * np.exp(-((t - c) ** 2) / (2 * 0.08 ** 2))
    acc2 = np.zeros((len(t), 3))
    acc2[:, 2] = 9.81 + sig
    scr2 = gt.phase_scramble(acc2, seed=4)

    def exkurt(x):
        x = x - x.mean()
        return (x ** 4).mean() / (x ** 2).mean() ** 2 - 3.0

    assert exkurt(acc2[:, 2]) > 5.0
    assert exkurt(scr2[:, 2]) < 1.5


def test_splice_transition():
    fs = 50.0
    a = make_sine_acc(1.0, fs, 10.0, amp=0.3)
    b = make_sine_acc(3.0, fs, 10.0, amp=4.0)
    sp = gt.splice(a, b, 0.4)  # transition at 4.0 s of 10 s
    assert sp.shape == a.shape
    # second part has much higher energy
    n = sp.shape[0]
    k = int(0.4 * n)
    assert gt.sma(sp[k:]) > 3 * gt.sma(sp[:k])


def test_rotation_preserves_magnitude():
    fs = 50.0
    acc = make_sine_acc(2.0, fs, 5.0)
    rot = gt.rotate(acc, seed=5)
    assert np.allclose(gt.magnitude(rot), gt.magnitude(acc), rtol=1e-9)
    assert not np.allclose(rot, acc)


def test_resample_preserves_freq():
    acc = make_sine_acc(2.0, 50.0, 10.0)
    rs = gt.resample_rate(acc, 50.0, 43.0)
    assert abs(gt.dominant_freq(rs, 43.0) - 2.0) <= 0.3


ALL_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    failed = 0
    for fn in ALL_TESTS:
        try:
            fn()
            print("PASS", fn.__name__)
        except AssertionError as e:
            failed += 1
            print("FAIL", fn.__name__, e)
    print(f"{len(ALL_TESTS) - failed}/{len(ALL_TESTS)} passed")
    sys.exit(1 if failed else 0)
