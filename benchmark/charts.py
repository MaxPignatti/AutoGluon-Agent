"""Benchmark figures for the README and posts, from a scored ``results.csv``.

    python benchmark/charts.py docs/benchmark-60min-halfmachine-2026-11-07-run2/results.csv docs/images

Writes, in light and dark variants:
- ``error-reduction-{light,dark}.png``: AutoGluon-Agent's relative error reduction vs AutoGluon, per dataset;
- ``promised-vs-delivered-{light,dark}.png``: each system's own validation score vs its test score on the
  leakage-track datasets (production variant: the leaky column is blank at prediction time).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score import relative_error_reduction  # noqa: E402

THEMES = {  # validated two-series palette (blue = AutoGluon-Agent, orange = AutoGluon) + chart chrome
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9",
                  axis="#c3c2b7", agent="#2a78d6", ag="#eb6834"),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a",
                 axis="#383835", agent="#3987e5", ag="#d95926"),
}
LEAK_DATASETS = {"bank-marketing": "bank-marketing (leak: call duration)",
                 "adult-planted-leak": "adult (leak: planted tax column)"}
FONT = ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"]


def _style(ax, t):
    ax.set_facecolor(t["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.tick_params(colors=t["muted"], length=0, labelsize=11)
    ax.grid(axis="x", color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def _title(fig, t, title, subtitle):
    fig.text(0.04, 0.95, title, fontsize=19, fontweight="bold", color=t["ink"], va="top")
    fig.text(0.04, 0.885, subtitle, fontsize=12, color=t["ink2"], va="top")


def error_reduction(df: pd.DataFrame, t: dict, out: Path) -> None:
    prod = df[df["variant"] == "production"]
    rows = []
    for ds, g in prod.groupby("dataset"):
        ag = g[g["system"] == "autogluon"].iloc[0]
        aga = g[g["system"] == "aga"].iloc[0]
        rows.append((ds, ag["metric"], 100 * relative_error_reduction(ag["metric"], ag["score"], aga["score"])))
    rows.sort(key=lambda r: r[2])
    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=150)
    fig.patch.set_facecolor(t["surface"])
    fig.subplots_adjust(left=0.24, right=0.93, top=0.78, bottom=0.12)
    _style(ax, t)
    labels = [f"{ds}\n{metric.upper().replace('_', ' ')}" for ds, metric, _ in rows]
    values = [v for _, _, v in rows]
    bars = ax.barh(labels, values, height=0.56, color=t["agent"])
    for bar in bars:  # 4px-style rounded data end is not native in matplotlib; keep plain thin bars
        bar.set_linewidth(0)
    for y, v in enumerate(values):
        ax.text(v + 0.15, y, f"+{v:.1f}%", va="center", ha="left", fontsize=12, color=t["ink"], fontweight="bold")
    ax.set_xlim(0, max(values) * 1.18)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0f}%")
    ax.tick_params(axis="y", labelsize=12, labelcolor=t["ink2"])
    ax.set_xlabel("Relative error reduction vs AutoGluon best_quality (higher is better)", color=t["muted"], fontsize=11)
    _title(fig, t, "AutoGluon-Agent beat AutoGluon on 6 of 6 datasets",
           "Same 60-minute budget, test set at prediction time. Error = 1 − AUC or RMSE. Single seed.")
    fig.savefig(out, facecolor=t["surface"])
    plt.close(fig)


def promised_vs_delivered(df: pd.DataFrame, t: dict, out: Path) -> None:
    prod = df[(df["variant"] == "production") & df["dataset"].isin(LEAK_DATASETS)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.75), dpi=150, sharey=True)
    fig.patch.set_facecolor(t["surface"])
    fig.subplots_adjust(left=0.17, right=0.97, top=0.74, bottom=0.2, wspace=0.12)
    systems = [("autogluon", "AutoGluon", t["ag"]), ("aga", "AutoGluon-Agent", t["agent"])]
    for ax, (ds, name) in zip(axes, LEAK_DATASETS.items()):
        _style(ax, t)
        g = prod[prod["dataset"] == ds]
        for y, (sys_id, label, color) in enumerate(systems):
            r = g[g["system"] == sys_id].iloc[0]
            promised, delivered = r["cv_score"], r["score"]
            gap = delivered - promised
            if abs(gap) < 0.006:  # points would overlap: concentric ring + dot reads as "promised = delivered"
                ax.scatter([delivered], [y], s=300, facecolor="none", edgecolor=color, linewidth=2, zorder=3)
                ax.scatter([delivered], [y], s=90, color=color, linewidth=0, zorder=4)
            else:
                ax.plot([promised, delivered], [y, y], color=color, linewidth=2, solid_capstyle="round", zorder=2)
                ax.scatter([promised], [y], s=110, facecolor=t["surface"], edgecolor=color, linewidth=2, zorder=3)
                ax.scatter([delivered], [y], s=110, color=color, edgecolor=t["surface"], linewidth=2, zorder=4)
            if abs(gap) > 0.02:
                ax.text((promised + delivered) / 2, y + 0.2, f"{gap:+.2f} AUC", ha="center", va="bottom",
                        fontsize=12, color=t["ink"], fontweight="bold")
            else:
                ax.text(max(promised, delivered) + 0.012, y, f"{gap:+.3f}", ha="left", va="center",
                        fontsize=12, color=t["ink"], fontweight="bold")
        ax.set_title(name, color=t["ink2"], fontsize=12.5, loc="left", pad=12)
        ax.set_ylim(-0.6, 1.7)
        ax.set_xlim(0.76, 1.0)
        ax.xaxis.set_major_formatter(lambda x, _: f"{x:.2f}")
        ax.set_yticks([0, 1], [s[1] for s in systems])
        ax.tick_params(axis="y", labelsize=12, labelcolor=t["ink2"])
    axes[0].set_xlabel("ROC AUC", color=t["muted"], fontsize=11)
    axes[1].set_xlabel("ROC AUC", color=t["muted"], fontsize=11)
    # legend for the two marks (identity is also in the y labels)
    for marker_face, text, x in ((t["surface"], "own validation score (promised)", 0.17),
                                 (t["muted"], "score on the test set (delivered)", 0.47)):
        fig.lines.append(plt.Line2D([x], [0.06], marker="o", markersize=10, markerfacecolor=marker_face,
                                    markeredgecolor=t["muted"], markeredgewidth=2, linestyle="",
                                    transform=fig.transFigure))
        fig.text(x + 0.015, 0.06, text, va="center", fontsize=11.5, color=t["ink2"])
    _title(fig, t, "AutoGluon's validation was fooled by the leak. The agent's was not.",
           "Leaky column blank at prediction time. AutoGluon-Agent found and dropped it before training.")
    fig.savefig(out, facecolor=t["surface"])
    plt.close(fig)


def main(argv: list[str]) -> int:
    csv, outdir = Path(argv[0]), Path(argv[1])
    outdir.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = FONT
    df = pd.read_csv(csv)
    for mode, t in THEMES.items():
        error_reduction(df, t, outdir / f"error-reduction-{mode}.png")
        promised_vs_delivered(df, t, outdir / f"promised-vs-delivered-{mode}.png")
    print(f"wrote 4 charts to {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
