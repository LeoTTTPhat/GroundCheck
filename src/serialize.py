"""Signal -> text serialization for LLM prompts (HARGPT-style).

Downsample the triaxial signal to `target_fs` (anti-aliased) and round to
`decimals`. Per-axis serialization (not magnitude) so that rotation-based
contamination probes remain meaningful. Deterministic.
"""
import numpy as np
from scipy import signal as sps


def downsample(acc, fs, target_fs):
    acc = np.asarray(acc, dtype=float)
    if abs(fs - target_fs) < 1e-9:
        return acc, fs
    from fractions import Fraction
    fr = Fraction(target_fs / fs).limit_denominator(100)
    out = sps.resample_poly(acc, fr.numerator, fr.denominator, axis=0)
    return out, fs * fr.numerator / fr.denominator


def serialize(acc, fs, target_fs=10.0, decimals=2, max_samples=120):
    """Return (text, ds_acc, ds_fs). Text lists x/y/z at ds_fs, rounded."""
    ds, ds_fs = downsample(acc, fs, target_fs)
    if ds.shape[0] > max_samples:
        ds = ds[:max_samples]
    dur = ds.shape[0] / ds_fs
    fmt = f"%.{decimals}f"
    lines = [f"Sampling rate: {ds_fs:g} Hz, duration: {dur:g} s, "
             f"{ds.shape[0]} samples per axis."]
    for j, name in enumerate("xyz"):
        vals = " ".join(fmt % v for v in ds[:, j])
        lines.append(f"{name}: {vals}")
    return "\n".join(lines), ds, ds_fs


def serialized_labels(labels, fs, target_fs=10.0, max_samples=240):
    """Downsample a per-sample label array by decimation (labels are not
    filtered). Used only to keep ground-truth label grids aligned with the
    serialized signal for transition items."""
    if labels is None:
        return None
    labels = np.asarray(labels)
    step = max(1, int(round(fs / target_fs)))
    out = labels[::step]
    return out[:max_samples]


# --------------------------------------------- Stage-3 format robustness --
# The audit's negative result must not be an artefact of ONE serialization.
# These variants re-render the SAME window in formats a referee would ask
# for; ground truth is unchanged (they are lossless-or-better re-renderings
# of the same underlying window), so accuracy is directly comparable.

def serialize_magnitude(acc, fs, target_fs=10.0, decimals=2,
                        max_samples=120):
    """Single-channel |a| series instead of three axes (3x fewer numbers).

    Removes the 'the model must combine three axes' confound: every family
    except intensity_order is a function of the magnitude alone.
    """
    ds, ds_fs = downsample(acc, fs, target_fs)
    if ds.shape[0] > max_samples:
        ds = ds[:max_samples]
    mag = np.sqrt((ds ** 2).sum(axis=1))
    dur = ds.shape[0] / ds_fs
    fmt = f"%.{decimals}f"
    return ("\n".join([
        f"Sampling rate: {ds_fs:g} Hz, duration: {dur:g} s, "
        f"{ds.shape[0]} samples.",
        "magnitude: " + " ".join(fmt % v for v in mag)]), ds, ds_fs)


def serialize_indexed(acc, fs, target_fs=10.0, decimals=2, max_samples=120):
    """'t=0.0 m=1.02' one timestamped magnitude sample per line.

    Tests whether explicit time indexing (rather than a bare number run)
    helps the model localise events; costs ~3x the tokens of the bare form.
    """
    ds, ds_fs = downsample(acc, fs, target_fs)
    if ds.shape[0] > max_samples:
        ds = ds[:max_samples]
    mag = np.sqrt((ds ** 2).sum(axis=1))
    fmt = f"%.{decimals}f"
    lines = [f"Sampling rate: {ds_fs:g} Hz, {ds.shape[0]} samples "
             f"(t seconds, m = |acceleration|)."]
    lines += [f"t={i / ds_fs:.1f} m=" + fmt % v for i, v in enumerate(mag)]
    return "\n".join(lines), ds, ds_fs


def serialize_coarse(acc, fs, target_fs=5.0, decimals=1, max_samples=60):
    """Half the rate, one decimal: a 4x shorter prompt.

    If accuracy is unchanged by removing 3/4 of the numbers, the model was
    not reading them.
    """
    return serialize_magnitude(acc, fs, target_fs=target_fs,
                               decimals=decimals, max_samples=max_samples)


SERIALIZERS = {
    "triaxial": serialize,          # the Stage-2 default
    "magnitude": serialize_magnitude,
    "indexed": serialize_indexed,
    "coarse": serialize_coarse,
}


def alt_formats_from_text(text, fmts=("magnitude", "indexed", "coarse")):
    """Re-render an already-serialized window in alternative text formats.

    Deriving the variants from the SERIALIZED text (not the raw signal)
    guarantees they carry no more information than the default triaxial
    rendering, so any accuracy difference is a pure format effect. Handles
    'Window A:/Window B:' two-window items.
    """
    import re as _re

    def _parse(t):
        m = _re.search(r"Sampling rate:\s*([\d.]+)\s*Hz", t)
        fs = float(m.group(1)) if m else 10.0
        ax = []
        for name in "xyz":
            mm = _re.search(rf"^{name}:\s*(.+)$", t, flags=_re.M)
            if not mm:
                return None, fs
            ax.append(np.array([float(v) for v in mm.group(1).split()]))
        n = min(len(a) for a in ax)
        return np.stack([a[:n] for a in ax], axis=1), fs

    def _one(t, fmt):
        acc, fs = _parse(t)
        if acc is None:
            return t
        return SERIALIZERS[fmt](acc, fs, target_fs=(5.0 if fmt == "coarse"
                                                    else fs),
                                decimals=(1 if fmt == "coarse" else 2),
                                max_samples=(60 if fmt == "coarse"
                                             else acc.shape[0]))[0]

    out = {}
    for fmt in fmts:
        if "Window B:" in text:
            a, b = text.split("Window B:")
            a = a.replace("Window A:", "").strip()
            out[fmt] = f"Window A:\n{_one(a, fmt)}\nWindow B:\n{_one(b.strip(), fmt)}"
        else:
            out[fmt] = _one(text, fmt)
    return out
