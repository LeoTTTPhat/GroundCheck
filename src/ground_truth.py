"""GroundCheck ground-truth engine.

Programmatic (numpy/scipy) oracle answers for every quantity family. No LLM
or human judgment anywhere in this file. All functions operate on raw numpy
arrays; sampling rate is always passed explicitly.

Families:
  peak_count      -- movement-cycle (step/repetition) peaks in accel magnitude
  dominant_freq   -- dominant frequency (Hz) of detrended magnitude (Welch PSD)
  intensity_order -- which of two windows has higher signal magnitude area
  transition_time -- time (s) of the first activity-label change
  bout_duration   -- duration (s) of the longest contiguous bout of a label
"""
import numpy as np
from scipy import signal as sps

GRAVITY = 9.80665


def magnitude(acc):
    """Euclidean norm per sample of an (N,3) acceleration array."""
    acc = np.asarray(acc, dtype=float)
    return np.sqrt((acc ** 2).sum(axis=1))


def _detrend_mag(mag):
    """Remove the (gravity/DC) mean so the oscillatory component remains."""
    mag = np.asarray(mag, dtype=float)
    return mag - mag.mean()


def bandpass_mag(mag, fs, lo=0.5, hi=5.0, order=3):
    """Band-pass the detrended magnitude to the human-movement band."""
    nyq = fs / 2.0
    hi = min(hi, 0.95 * nyq)
    if lo >= hi:  # pathological fs; fall back to detrend only
        return _detrend_mag(mag)
    b, a = sps.butter(order, [lo / nyq, hi / nyq], btype="band")
    return sps.filtfilt(b, a, _detrend_mag(mag))


def count_peaks(acc, fs, min_freq=0.5, max_freq=5.0):
    """Count movement-cycle peaks in the filtered acceleration magnitude.

    Peaks = local maxima with prominence >= 0.3 * std of the filtered signal
    and spacing >= 1/max_freq seconds. On periodic gait signals this counts
    one peak per step-cycle impact of the dominant oscillation.
    """
    mag = magnitude(acc) if np.ndim(acc) == 2 else np.asarray(acc, float)
    filt = bandpass_mag(mag, fs, lo=min_freq, hi=max_freq)
    sd = filt.std()
    if sd < 1e-12:
        return 0
    dist = max(1, int(round(fs / max_freq)))
    peaks, _ = sps.find_peaks(filt, prominence=0.3 * sd, distance=dist)
    return int(len(peaks))


def dominant_freq(acc, fs, min_freq=0.3, max_freq=10.0):
    """Dominant frequency (Hz) of the detrended magnitude via Welch PSD."""
    mag = magnitude(acc) if np.ndim(acc) == 2 else np.asarray(acc, float)
    x = _detrend_mag(mag)
    nper = min(len(x), max(64, int(4 * fs)))
    f, pxx = sps.welch(x, fs=fs, nperseg=nper)
    band = (f >= min_freq) & (f <= min(max_freq, 0.99 * fs / 2))
    if not band.any() or pxx[band].max() < 1e-15:
        return 0.0
    return float(f[band][np.argmax(pxx[band])])


def sma(acc, fs=None):
    """Signal magnitude area (gravity-removed, per-axis high-passed mean abs).

    SMA = mean over samples of |x_hp| + |y_hp| + |z_hp| where _hp removes the
    per-axis mean (static gravity component). Units follow the input units.
    """
    acc = np.asarray(acc, dtype=float)
    hp = acc - acc.mean(axis=0, keepdims=True)
    return float(np.abs(hp).sum(axis=1).mean())


def intensity_order(acc_a, acc_b, fs=None):
    """Return 'A' if window A has strictly higher SMA else 'B'."""
    return "A" if sma(acc_a) > sma(acc_b) else "B"


def transition_time(labels, fs):
    """Time (s, center of the boundary) of the FIRST label change; None if none."""
    labels = np.asarray(labels)
    change = np.nonzero(labels[1:] != labels[:-1])[0]
    if len(change) == 0:
        return None
    # boundary between sample i and i+1 -> time (i+1)/fs
    return float((change[0] + 1) / fs)


def all_transition_times(labels, fs):
    labels = np.asarray(labels)
    change = np.nonzero(labels[1:] != labels[:-1])[0]
    return [float((i + 1) / fs) for i in change]


def bout_duration(labels, fs, target):
    """Duration (s) of the LONGEST contiguous run of `target` in labels."""
    labels = np.asarray(labels)
    mask = (labels == target).astype(int)
    if mask.sum() == 0:
        return 0.0
    # run-length encode
    edges = np.diff(np.concatenate([[0], mask, [0]]))
    starts = np.nonzero(edges == 1)[0]
    ends = np.nonzero(edges == -1)[0]
    return float((ends - starts).max() / fs)


# ---------------------------------------------------------------- edits ----

def phase_scramble(acc, seed=0):
    """Phase-randomize each axis: keep FFT magnitude, randomize phases.

    Destroys temporal structure (peak trains, transitions) while preserving
    the power spectrum and variance.
    """
    rng = np.random.default_rng(seed)
    acc = np.asarray(acc, dtype=float)
    out = np.empty_like(acc)
    for j in range(acc.shape[1]):
        X = np.fft.rfft(acc[:, j])
        ph = rng.uniform(0, 2 * np.pi, size=len(X))
        ph[0] = 0.0  # keep DC real
        Xs = np.abs(X) * np.exp(1j * ph)
        out[:, j] = np.fft.irfft(Xs, n=acc.shape[0])
    return out


def time_stretch(acc, factor):
    """Resample so events take `factor` x as long (factor>1 -> slower cadence).

    Output has the same number of samples (crops/pads by resampling a
    stretched copy back onto the original grid), so dominant frequency is
    divided by `factor`.
    """
    acc = np.asarray(acc, dtype=float)
    n = acc.shape[0]
    n_stretch = int(round(n * factor))
    stretched = sps.resample(acc, n_stretch, axis=0)
    if n_stretch >= n:
        return stretched[:n]
    reps = int(np.ceil(n / n_stretch))
    return np.tile(stretched, (reps, 1))[:n]


def splice(acc_a, acc_b, frac):
    """Concatenate the first frac of A with the tail of B (same total length)."""
    acc_a = np.asarray(acc_a, dtype=float)
    acc_b = np.asarray(acc_b, dtype=float)
    n = min(acc_a.shape[0], acc_b.shape[0])
    k = int(round(n * frac))
    return np.concatenate([acc_a[:k], acc_b[k:n]], axis=0)


def random_rotation(seed=0):
    """Uniformly random 3D rotation matrix (QR of a Gaussian matrix)."""
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((3, 3))
    q, r = np.linalg.qr(m)
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def rotate(acc, seed=0):
    return np.asarray(acc, dtype=float) @ random_rotation(seed).T


def resample_rate(acc, fs_in, fs_out):
    """Resample to a different sampling rate (anti-aliased polyphase)."""
    from fractions import Fraction
    fr = Fraction(fs_out / fs_in).limit_denominator(100)
    return sps.resample_poly(np.asarray(acc, float), fr.numerator,
                             fr.denominator, axis=0)
