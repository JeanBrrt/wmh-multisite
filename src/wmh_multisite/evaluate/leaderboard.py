"""Position of our methods on the official WMH challenge leaderboard (ROADMAP step 6.4, J-042, J-060).

Source: the leaderboard of the challenge (57 teams, up to date in December 2022), page 20 of the
official ``readme.pdf`` of the dataset (DataverseNL), stored in ``results/tables/leaderboard_wmh2017.csv``
(values rounded to 2 decimals in the source).

Official ranking (``readme.pdf``, page 23): each metric is averaged over the test scans; for each metric
the best team gets 0, the worst 1, the others linearly in between; the overall rank is the mean of
the five metric ranks (Dice, HD95, AVD, lesion recall, lesion F1). Adding a team can change the best or
worst value of a metric, hence every team's score: each of our methods is inserted **alone** among
the 57 teams and ranked with the same formula.
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
