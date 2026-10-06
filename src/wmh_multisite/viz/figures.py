"""Summary figures of the evaluation (ROADMAP step 6.4, J-060).

    results/figures/
      dice_by_method_and_scanner.png   mean Dice and bootstrap 95 % CI (6.2) per method and scanner
      seen_vs_unseen.png               Dice and HD95, scanners seen vs unseen in training, with CIs
      leaderboard.png                  official overall rank score of the 57 published teams + ours (6.4)
      human_ceiling.png                inter-observer agreement (6.3) next to the methods
    results/tables/leaderboard_positions.csv

Only aggregated numbers are drawn (no patient image).

Usage:
    uv run python -m wmh_multisite.viz.figures
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from wmh_multisite.config import load_config, resolve_path  # noqa: E402
from wmh_multisite.evaluate.leaderboard import HIGHER_IS_BETTER, position, ranking  # noqa: E402

log = logging.getLogger(__name__)

METHODS = {  # key -> label (main comparison; variants are in the tables)
    "resencm": "M1 nnU-Net",
    "resencm_da5": "M1 + DA5",
    "threshold": "M3 threshold",
    "wmhsynthseg": "M2 WMH-SynthSeg",
}
COLORS = {"resencm": "#1f77b4", "resencm_da5": "#7fb3e0", "threshold": "#ff7f0e", "wmhsynthseg": "#2ca02c"}


def _ci(ci: pd.DataFrame, model: str, group: str, metric: str) -> tuple[float, float, float]:
    r = ci[(ci.model == model) & (ci.group == group) & (ci.metric == metric)].iloc[0]
    return r["mean"], r["mean"] - r.ci_low, r.ci_high - r["mean"]


def dice_by_method_and_scanner(ci: pd.DataFrame, path: Path) -> None:
    groups = ["all"] + sorted(g for g in ci.group.unique() if g.startswith("scanner="))
    models = [m for m in METHODS if m in set(ci.model)]
    x = np.arange(len(groups))
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for i, m in enumerate(models):
        vals = [_ci(ci, m, g, "dice") for g in groups]
        ax.bar(
            x + i * width,
            [v[0] for v in vals],
            width,
            yerr=np.array([[v[1] for v in vals], [v[2] for v in vals]]),
            capsize=2,
            label=METHODS[m],
            color=COLORS[m],
        )
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels([g.replace("scanner=", "").replace("_", "\n") for g in groups], fontsize=8)
    ax.set_ylabel("Dice (mean, 95 % bootstrap CI)")
    ax.set_ylim(0, 1)
    ax.set_title("Test set (110 scans): Dice per method and scanner", fontsize=10)
    ax.legend(fontsize=8, ncol=4, loc="upper center")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def seen_vs_unseen(ci: pd.DataFrame, path: Path) -> None:
    models = [m for m in METHODS if m in set(ci.model)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric, label in [(axes[0], "dice", "Dice"), (axes[1], "hd95_mm", "HD95 (mm)")]:
        for j, (group, name) in enumerate([("seen=True", "seen (90)"), ("seen=False", "unseen (20)")]):
            vals = [_ci(ci, m, group, metric) for m in models]
            x = np.arange(len(models)) + (j - 0.5) * 0.35
            ax.bar(
                x,
                [v[0] for v in vals],
                0.35,
                yerr=np.array([[v[1] for v in vals], [v[2] for v in vals]]),
                capsize=3,
                label=name,
                color=["#555555", "#bbbbbb"][j],
            )
        ax.set_xticks(np.arange(len(models)))
        ax.set_xticklabels([METHODS[m] for m in models], fontsize=8, rotation=15)
        ax.set_ylabel(f"{label} (mean, 95 % CI)")
        ax.legend(fontsize=8)
    fig.suptitle("Scanners seen vs unseen in training (test set)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def leaderboard_figure(leaderboard: pd.DataFrame, ours: dict[str, dict[str, Any]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.2))
    published = ranking(leaderboard).sort_values().to_numpy()
    ax.plot(
        np.arange(1, len(published) + 1), published, "o", color="#999999", ms=4, label="57 published teams"
    )
    for model, p in ours.items():
        ax.plot(
            p["position"],
            p["score"],
            "*",
            ms=14,
            color=COLORS[model],
            label=f"{METHODS[model]}: {p['position']}/{p['n_teams']}",
        )
    ax.set_xlabel("position")
    ax.set_ylabel("official overall rank score (lower is better)")
    ax.set_title(
        "MICCAI WMH 2017 leaderboard (readme.pdf, Dec. 2022) with our methods inserted one at a time",
        fontsize=9,
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def human_ceiling(inter: pd.DataFrame, ci: pd.DataFrame, path: Path) -> None:
    rows = [
        ("O3 vs O1 (60 train)", inter.loc["O3 vs O1 | all", "dice"], "#8c564b"),
        ("O4 vs O1 (60 train)", inter.loc["O4 vs O1 | all", "dice"], "#8c564b"),
        ("O4 vs O3 (60 train)", inter.loc["O4 vs O3 | all", "dice"], "#c49c94"),
        ("O3 vs O1 (12 shared)", inter.loc["O3 vs O1 | 12 scans shared with M1", "dice"], "#8c564b"),
        ("O4 vs O1 (12 shared)", inter.loc["O4 vs O1 | 12 scans shared with M1", "dice"], "#8c564b"),
        ("M1 vs O1 (12 shared)", inter.loc["M1 vs O1 | 12 scans shared with M1", "dice"], COLORS["resencm"]),
        ("M1 vs O1 (110 test)", _ci(ci, "resencm", "all", "dice")[0], COLORS["resencm"]),
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh([r[0] for r in rows], [r[1] for r in rows], color=[r[2] for r in rows])
    for i, r in enumerate(rows):
        ax.text(r[1] + 0.005, i, f"{r[1]:.3f}", va="center", fontsize=8)
    ax.set_xlim(0.6, 0.9)
    ax.invert_yaxis()
    ax.set_xlabel("Dice against the reference O1")
    ax.set_title("Human ceiling: inter-observer agreement vs nnU-Net (M1)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Summary figures (6.4).")
    parser.add_argument("--config")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    cfg = load_config(args.config)
    tables = resolve_path(cfg, "results") / "tables"
    figures = resolve_path(cfg, "results") / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    ci = pd.read_csv(tables / "stats_ci.csv")
    leaderboard = pd.read_csv(tables / "leaderboard_wmh2017.csv")
    ours = {}
    for model in METHODS:
        f = tables / f"evaluation_{model}.csv"
        if f.is_file():
            e = pd.read_csv(f)
            ours[model] = position(leaderboard, {m: float(e[m].mean()) for m in HIGHER_IS_BETTER}, model)
    pd.DataFrame(ours).T.rename_axis("model").to_csv(
        tables / "leaderboard_positions.csv", float_format="%.4g"
    )
    dice_by_method_and_scanner(ci, figures / "dice_by_method_and_scanner.png")
    seen_vs_unseen(ci, figures / "seen_vs_unseen.png")
    leaderboard_figure(leaderboard, ours, figures / "leaderboard.png")
    human_ceiling(
        pd.read_csv(tables / "interobserver_summary.csv", index_col=0), ci, figures / "human_ceiling.png"
    )
    log.info("leaderboard positions: %s", {m: f"{p['position']}/{p['n_teams']}" for m, p in ours.items()})


if __name__ == "__main__":
    main()
