"""The three figures in DESIGN.md section 8."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COLOURS = {"native": "#444444", "historical": "#1f77b4", "independent": "#d62728", "cross": "#2ca02c"}


def _summaries(period_dir: Path) -> dict[str, dict]:
    out = {}
    for f in sorted(period_dir.glob("*/summary.json")):
        label = f.parent.name
        if "_" in label:
            continue
        s = json.loads(f.read_text())
        if s.get("n_cases"):
            out[label] = s
    return out


def crps_by_coupling(summaries: dict[str, dict], ax) -> None:
    systems = list(summaries)
    width = 0.8 / 4
    for i, system in enumerate(systems):
        d = summaries[system]["differences_vs_native"]["crps"]
        for j, c in enumerate([c for c in ("historical", "independent", "cross") if c in d]):
            r = d[c]
            x = i + (j - 1) * width
            ax.errorbar(x, r["estimate"], yerr=[[r["estimate"] - r["lo"]], [r["hi"] - r["estimate"]]],
                        fmt="o", color=COLOURS[c], capsize=4, label=c if i == 0 else None)
    ax.axhline(0, color=COLOURS["native"], lw=1)
    ax.set_xticks(range(len(systems)), [s.upper() for s in systems])
    ax.set_ylabel("Portfolio CRPS minus native\n(fraction of portfolio capacity)")
    ax.set_title("Portfolio CRPS by coupling\n95% block bootstrap interval", fontsize=11)
    ax.legend(frameon=False)


def aggregation_curve(summaries: dict[str, dict], axes) -> None:
    for ax, (system, s) in zip(np.atleast_1d(axes), summaries.items(), strict=False):
        sizes = sorted(int(k) for k in s["aggregation"])
        for c in ("historical", "independent", "cross"):
            if c not in s["aggregation"][str(sizes[0])]:
                continue
            est = np.array([s["aggregation"][str(k)][c]["estimate"] for k in sizes])
            lo = np.array([s["aggregation"][str(k)][c]["lo"] for k in sizes])
            hi = np.array([s["aggregation"][str(k)][c]["hi"] for k in sizes])
            ax.plot(sizes, est, "-o", ms=3, color=COLOURS[c], label=c)
            ax.fill_between(sizes, lo, hi, color=COLOURS[c], alpha=0.15)
        ax.axhline(0, color=COLOURS["native"], lw=1)
        ax.set_xlabel("Farms aggregated")
        ax.set_title(system.upper())
    np.atleast_1d(axes)[0].set_ylabel("CRPS minus native\n(fraction of subset capacity)")
    np.atleast_1d(axes)[0].legend(frameon=False)


def rank_histograms(summaries: dict[str, dict], axes) -> None:
    for ax, (system, s) in zip(np.atleast_1d(axes), summaries.items(), strict=False):
        for c in ("native", "historical"):
            h = np.array(s["rank_histogram"][c], dtype=float)
            h = h / h.sum() * len(h)
            ax.step(np.arange(len(h)), h, where="mid", color=COLOURS[c], label=c)
        ax.axhline(1, color="#999999", lw=0.8, ls=":")
        ax.set_xlabel("Rank of observed portfolio output")
        ax.set_title(system.upper())
    np.atleast_1d(axes)[0].set_ylabel("Relative frequency (1 = uniform)")
    np.atleast_1d(axes)[0].legend(frameon=False)


def make_all(period_dir: Path, synthetic: bool = False) -> list[Path]:
    summaries = _summaries(period_dir)
    if not summaries:
        raise FileNotFoundError(f"no summaries under {period_dir}")
    outdir = period_dir / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    tag = " [SYNTHETIC]" if synthetic else ""
    n = len(summaries)
    files = []

    fig, ax = plt.subplots(figsize=(6, 4))
    crps_by_coupling(summaries, ax)
    ax.set_title(ax.get_title() + tag)
    files.append(outdir / "fig1_crps_by_coupling.png")

    fig2, axes2 = plt.subplots(1, n, figsize=(5 * n, 3.8), sharey=True)
    aggregation_curve(summaries, axes2)
    fig2.suptitle("Effect of coupling by aggregation level" + tag)
    files.append(outdir / "fig2_aggregation.png")

    fig3, axes3 = plt.subplots(1, n, figsize=(5 * n, 3.8), sharey=True)
    rank_histograms(summaries, axes3)
    fig3.suptitle("Portfolio rank histogram, native against historical" + tag)
    files.append(outdir / "fig3_rank_histogram.png")

    for f, path in zip((fig, fig2, fig3), files, strict=True):
        f.tight_layout()
        f.savefig(path, dpi=150)
        plt.close(f)
    return files
