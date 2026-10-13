"""Position of our methods on the official WMH challenge leaderboard.

Source: the 57-team leaderboard (December 2022) on page 20 of the dataset's ``readme.pdf``
(DataverseNL), copied to ``results/tables/leaderboard_wmh2017.csv`` (rounded to 2 decimals there).

Official ranking (``readme.pdf``, page 23): for each metric averaged over the test scans, the best team
gets 0, the worst 1, the others linearly in between; the overall rank is the mean over the five
metrics. A new team can move the best or worst value and so every score, hence each of our methods is
inserted alone among the 57 teams.
"""

from __future__ import annotations

import pandas as pd

HIGHER_IS_BETTER = {
    "dice": True,
    "hd95_mm": False,
    "avd_pct": False,
    "lesion_recall": True,
    "lesion_f1": True,
}


def ranking(teams: pd.DataFrame) -> pd.Series:
    """Official overall rank score of each row (lower is better), from the mean metrics."""
    score = pd.Series(0.0, index=teams.index)
    for metric, higher in HIGHER_IS_BETTER.items():
        v = teams[metric]
        r = (v - v.min()) / (v.max() - v.min())
        score += (1 - r) if higher else r
    return score / len(HIGHER_IS_BETTER)


def position(leaderboard: pd.DataFrame, ours: dict[str, float], name: str) -> dict[str, float]:
    """Insert one method (mean metrics) among the published teams; its position and per-metric ranks."""
    teams = pd.concat(
        [leaderboard[["team", *HIGHER_IS_BETTER]], pd.DataFrame([{"team": name, **ours}])], ignore_index=True
    )
    teams["score"] = ranking(teams)
    teams = teams.sort_values("score").reset_index(drop=True)
    out = {
        "position": int(teams.index[teams.team == name][0]) + 1,
        "n_teams": len(teams),
        "score": float(teams.loc[teams.team == name, "score"].iloc[0]),
    }
    for metric, higher in HIGHER_IS_BETTER.items():
        order = teams[metric].rank(ascending=not higher, method="min")
        out[f"rank_{metric}"] = int(order[teams.team == name].iloc[0])
    return out
