#!/usr/bin/env python3
"""Figures for the paper, generated from results/metrics.json only.

Writes paper/figures/{fig_scale,fig_accuracy,fig_prior_gap,fig_grounding,
fig_decomposition,fig_cost}.pdf.  Error bars are the Wilson / bootstrap
intervals already stored in metrics.json -- nothing is recomputed here.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")
FAMILIES = ["peak_count", "dominant_freq", "intensity_order",
            "transition_time", "bout_duration"]
FAM_LAB = ["peak\ncount", "dominant\nfreq", "intensity\norder",
           "transition\ntime", "bout\nduration"]
GRID_B = ["no_signal", "with_signal", "with_signal_cot", "with_signal_long",
          "feature_prompt", "tool_forced", "tool_augmented"]
MAIN_B = ["no_signal", "with_signal", "with_signal_cot", "tool_augmented"]
B_LAB = {"no_signal": "no-signal (prior)", "with_signal": "with-signal",
         "with_signal_cot": "with-signal + CoT",
         "with_signal_long": "with-signal (128-tok cap)",
         "feature_prompt": "features-in-prompt",
         "tool_forced": "tool (forced routing)",
         "tool_augmented": "tool-augmented"}
COLORS = {"no_signal": "#9e9e9e", "with_signal": "#4878cf",
          "with_signal_cot": "#6acc65", "with_signal_long": "#8ab7e8",
          "feature_prompt": "#b47cc7", "tool_forced": "#ee9944",
          "tool_augmented": "#d65f5f"}

plt.rcParams.update({"font.size": 8, "axes.titlesize": 9,
                     "axes.labelsize": 8, "legend.fontsize": 7,
                     "pdf.fonttype": 42, "figure.dpi": 150})


def models_of(M):
    return [m for m in M["grid"]]


def short(M, m):
    return M["grid"][m]["short"]


def _err(entries):
    """(2,N) yerr array from stored CI bounds."""
    lo = [e["acc"] - e["ci_lo"] for e in entries]
    hi = [e["ci_hi"] - e["acc"] for e in entries]
    return np.array([lo, hi])


# ------------------------------------------------------------ figure 1 ---

def fig_scale(M):
    """The headline: accuracy and prior-gap vs model size, all conditions.

    An evenly spaced categorical x-axis, not a log-parameter axis: 7B, 8B and
    9B are almost coincident in log-space and their tick labels collide.
    Spacing by rank keeps all six models legible and loses nothing, since the
    trend statistic is a rank correlation anyway.
    """
    ms = models_of(M)
    sizes = [M["grid"][m]["size_b"] for m in ms]
    order = sorted(range(len(ms)), key=lambda k: sizes[k])
    ms = [ms[k] for k in order]
    xs_all = list(range(len(ms)))
    labels = [short(M, m).replace("Qwen2.5-", "Qwen\n")
              .replace("Llama-3.1-", "Llama\n").replace("Gemma-2-", "Gemma\n")
              for m in ms]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for b in MAIN_B + ["tool_forced", "feature_prompt"]:
        xs, ys, el, eh = [], [], [], []
        for x, m in zip(xs_all, ms):
            e = M["grid"][m].get(b)
            if not e:
                continue
            xs.append(x)
            ys.append(e["overall"]["acc"])
            el.append(e["overall"]["acc"] - e["overall"]["ci_lo"])
            eh.append(e["overall"]["ci_hi"] - e["overall"]["acc"])
        if xs:
            axes[0].errorbar(xs, ys, yerr=[el, eh], marker="o", ms=3.5,
                             lw=1.3, capsize=2, color=COLORS[b],
                             label=B_LAB[b])
    axes[0].axhline(M["dsp_oracle"]["overall"]["acc"], color="black",
                    ls="--", lw=1.0, label="DSP-oracle ceiling")
    if "best_constant" in M:
        axes[0].axhline(M["best_constant"]["overall"], color="#b03060",
                        ls=":", lw=1.2, label="best constant (floor)")
    axes[0].set_ylabel("overall accuracy")
    axes[0].set_ylim(0, 1.02)
    axes[0].legend(fontsize=6, loc="upper center",
                   bbox_to_anchor=(0.5, -0.28), ncol=3, frameon=False)

    for b in ["with_signal", "with_signal_cot", "tool_augmented",
              "feature_prompt"]:
        xs, ys, el, eh = [], [], [], []
        for x, m in zip(xs_all, ms):
            g = M["prior_reliance_gap"].get(m, {}).get(b, {}).get("overall")
            if not g:
                continue
            xs.append(x)
            ys.append(g["gap"])
            el.append(g["gap"] - g["ci_lo"])
            eh.append(g["ci_hi"] - g["gap"])
        if xs:
            axes[1].errorbar(xs, ys, yerr=[el, eh], marker="o", ms=3.5,
                             lw=1.3, capsize=2, color=COLORS[b],
                             label=B_LAB[b])
    mg = M["equivalence_margin"]
    axes[1].axhspan(-mg, mg, color="#cccccc", alpha=0.45, lw=0,
                    label=f"$\\pm${mg} band")
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_ylabel("prior-reliance gap")
    axes[1].legend(fontsize=6, loc="upper center",
                   bbox_to_anchor=(0.5, -0.28), ncol=3, frameon=False)

    for ax in axes:
        ax.set_xticks(xs_all)
        ax.set_xticklabels(labels, fontsize=5.5)
        ax.set_xlim(-0.4, len(ms) - 0.6)
        ax.set_xlabel("model (increasing size)")
        ax.grid(alpha=0.3, lw=0.4)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.34)
    fig.savefig(os.path.join(FIG, "fig_scale.pdf"), bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------ figure 2 ---

def fig_accuracy(M):
    ms = models_of(M)
    n = len(ms)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7.0, 2.0 * rows),
                             sharey=True, squeeze=False)
    x = np.arange(len(FAMILIES))
    w = 0.2
    for k, m in enumerate(ms):
        ax = axes[k // cols][k % cols]
        for j, b in enumerate(MAIN_B):
            e = M["grid"][m].get(b)
            if not e:
                continue
            ent = [e["per_family"][f] for f in FAMILIES
                   if f in e["per_family"]]
            xs = [x[i] for i, f in enumerate(FAMILIES)
                  if f in e["per_family"]]
            ax.bar(np.array(xs) + (j - 1.5) * w,
                   [q["acc"] for q in ent], w, yerr=_err(ent),
                   error_kw={"lw": 0.5, "capsize": 1}, label=B_LAB[b],
                   color=COLORS[b], edgecolor="white", linewidth=0.3)
        dsp = [M["dsp_oracle"]["per_family"][f]["acc"] for f in FAMILIES]
        for xi, d in zip(x, dsp):
            ax.hlines(d, xi - 2 * w, xi + 2 * w, color="black", ls="--",
                      lw=0.9, label="DSP-oracle ceiling" if xi == 0 else None)
        ax.set_xticks(x)
        ax.set_xticklabels(FAM_LAB, fontsize=6)
        ax.set_title(short(M, m))
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3, lw=0.4)
        ax.set_axisbelow(True)
    for k in range(n, rows * cols):
        axes[k // cols][k % cols].axis("off")
    axes[0][0].set_ylabel("tolerance-band accuracy")
    axes[0][0].legend(fontsize=5.5, loc="upper right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_accuracy.pdf"), bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------ figure 3 ---

def fig_grounding(M):
    """The two counterfactual instruments, pooled, against oracle references."""
    ms = models_of(M)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.15))
    labels, hal, hal_e, cons, cons_e, cols = [], [], [], [], [], []
    for m in ms:
        for b in ["with_signal", "tool_augmented"]:
            h = M["hallucination"].get(f"{m}|{b}")
            c = M["cf_consistency"].get(f"{m}|{b}")
            if not h or not c or not h["pooled"].get("n"):
                continue
            labels.append(short(M, m).replace("Qwen2.5-", "Qwen-")
                          .replace("Llama-3.1-", "Llama-")
                          .replace("Gemma-2-", "Gemma-"))
            hal.append(h["pooled"]["rate"])
            hal_e.append([h["pooled"]["rate"] - h["pooled"]["ci_lo"],
                          h["pooled"]["ci_hi"] - h["pooled"]["rate"]])
            cons.append(c["pooled"]["consistency"])
            cons_e.append([c["pooled"]["consistency"] - c["pooled"]["ci_lo"],
                           c["pooled"]["ci_hi"] - c["pooled"]["consistency"]])
            cols.append(COLORS[b])
    ho = M["hallucination"].get("dsp_oracle|dsp_oracle")
    co = M["cf_consistency"].get("dsp_oracle|dsp_oracle")
    x = np.arange(len(labels))
    axes[0].bar(x, hal, 0.7, yerr=np.array(hal_e).T,
                error_kw={"lw": 0.6, "capsize": 1.5}, color=cols,
                edgecolor="white", lw=0.3)
    if ho:
        axes[0].axhline(ho["pooled"]["rate"], color="black", ls="--", lw=1.1)
    axes[0].set_ylabel("hallucination rate\n(lower is better)")
    axes[1].bar(x, cons, 0.7, yerr=np.array(cons_e).T,
                error_kw={"lw": 0.6, "capsize": 1.5}, color=cols,
                edgecolor="white", lw=0.3)
    if co:
        axes[1].axhline(co["pooled"]["consistency"], color="black", ls="--",
                        lw=1.1)
    axes[1].set_ylabel("directional consistency\n(higher is better)")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=5.5, rotation=35, ha="right")
        ax.set_ylim(0, 1.02)
        ax.grid(axis="y", alpha=0.3, lw=0.4)
        ax.set_axisbelow(True)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=COLORS["with_signal"]),
               plt.Rectangle((0, 0), 1, 1, fc=COLORS["tool_augmented"]),
               plt.Line2D([0], [0], color="black", ls="--", lw=1.1)]
    fig.legend(handles, ["with-signal", "tool-augmented",
                         "DSP-oracle reference"],
               fontsize=6.5, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.10))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_grounding.pdf"), bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------ figure 4 ---

def fig_decomposition(M):
    """The whole result in one place.

    Left: accuracy by condition for every model, read between the
    best-constant floor and the DSP-oracle ceiling.  Right: the same models'
    prior-reliance gaps with the equivalence band.  Merged from two earlier
    figures because the left panel of the scale plot duplicated this one.
    """
    ms = [m for m in models_of(M) if m in M["tool_decomposition"]]
    if not ms:
        return
    ms = sorted(ms, key=lambda m: M["grid"][m]["size_b"])
    labels = [short(M, m).replace("Qwen2.5-", "Qwen\n")
              .replace("Llama-3.1-", "Llama\n").replace("Gemma-2-", "Gemma\n")
              for m in ms]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.42),
                             gridspec_kw={"width_ratios": [1.45, 1]})
    x = np.arange(len(ms))

    # ---- left: accuracy by condition, floor and ceiling ----------------
    w = 0.2
    series = [("no_signal", "no-signal"),
              ("with_signal", "with-signal"),
              ("tool_augmented", "tool-augmented"),
              ("feature_prompt", "measurements")]
    for j, (b, lab) in enumerate(series):
        vals, errs = [], []
        for m in ms:
            e = M["grid"][m].get(b)
            vals.append(e["overall"]["acc"] if e else np.nan)
            errs.append([e["overall"]["acc"] - e["overall"]["ci_lo"],
                         e["overall"]["ci_hi"] - e["overall"]["acc"]]
                        if e else [0, 0])
        axes[0].bar(x + (j - 1.5) * w, vals, w, yerr=np.array(errs).T,
                    error_kw={"lw": 0.6, "capsize": 1.5}, label=lab,
                    color=COLORS[b], edgecolor="white", lw=0.3)
    axes[0].axhline(M["dsp_oracle"]["overall"]["acc"], color="black",
                    ls="--", lw=1.1, label="oracle ceiling")
    if "best_constant" in M:
        bc = M["best_constant"]
        # the deployable floor is the baseline; the in-sample one is a
        # concentration diagnostic, so both are drawn and distinguished
        if bc.get("deployable_overall") is not None:
            axes[0].axhline(bc["deployable_overall"], color="#b03060",
                            ls=":", lw=1.5,
                            label="floor (out-of-sample)")
            axes[0].axhline(bc["overall"], color="#b03060", ls=(0, (1, 3)),
                            lw=1.0, alpha=0.75,
                            label="floor (in-sample)")
        else:
            axes[0].axhline(bc["overall"], color="#b03060",
                            ls=":", lw=1.5, label="best constant (floor)")
    axes[0].set_ylabel("overall accuracy")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(fontsize=6, loc="upper center",
                   bbox_to_anchor=(0.5, -0.24), ncol=4, frameon=False,
                   columnspacing=0.9, handletextpad=0.4)

    # ---- right: prior-reliance gap -------------------------------------
    for b in ["with_signal", "tool_augmented", "feature_prompt"]:
        xs, ys, el, eh = [], [], [], []
        for xi, m in zip(x, ms):
            g = M["prior_reliance_gap"].get(m, {}).get(b, {}).get("overall")
            if not g:
                continue
            xs.append(xi)
            ys.append(g["gap"])
            el.append(g["gap"] - g["ci_lo"])
            eh.append(g["ci_hi"] - g["gap"])
        if xs:
            axes[1].errorbar(xs, ys, yerr=[el, eh], marker="o", ms=3.5,
                             lw=1.3, capsize=2, color=COLORS[b],
                             label=B_LAB[b])
    mg = M["equivalence_margin"]
    axes[1].axhspan(-mg, mg, color="#cccccc", alpha=0.5, lw=0,
                    label=f"$\\pm${mg} band")
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_ylabel("prior-reliance gap")
    axes[1].legend(fontsize=6, loc="upper center",
                   bbox_to_anchor=(0.5, -0.24), ncol=2, frameon=False)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=6)
        ax.grid(axis="y", alpha=0.3, lw=0.4)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.30)
    fig.savefig(os.path.join(FIG, "fig_decomposition.pdf"),
                bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------ figure 5 ---

def fig_cost(M):
    ms = models_of(M)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3))
    x = np.arange(len(ms))
    w = 0.16
    for j, b in enumerate(MAIN_B + ["tool_forced"]):
        lat = [M["grid"][m][b]["mean_latency_s"] if b in M["grid"][m]
               else np.nan for m in ms]
        tok = [M["grid"][m][b]["mean_prompt_tokens"] if b in M["grid"][m]
               else np.nan for m in ms]
        axes[0].bar(x + (j - 2) * w, lat, w, color=COLORS[b], label=B_LAB[b],
                    edgecolor="white", lw=0.3)
        axes[1].bar(x + (j - 2) * w, tok, w, color=COLORS[b],
                    edgecolor="white", lw=0.3)
    axes[0].set_ylabel("mean latency (s)")
    axes[1].set_ylabel("mean prompt tokens")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([short(M, m) for m in ms], fontsize=6,
                           rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.3, lw=0.4)
        ax.set_axisbelow(True)
    axes[0].legend(fontsize=5.5, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_cost.pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    with open(os.path.join(ROOT, "results", "metrics.json")) as f:
        M = json.load(f)
    for fn in (fig_scale, fig_accuracy, fig_grounding, fig_decomposition,
               fig_cost):
        try:
            fn(M)
            print("wrote", fn.__name__)
        except Exception as e:  # a section may be absent mid-run
            print(f"skip {fn.__name__}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
