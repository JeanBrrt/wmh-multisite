"""Official WMH challenge ranking (6.4) and the summary figures."""

import pandas as pd
import pytest

from wmh_multisite.evaluate.leaderboard import position, ranking


def teams():
    # readme.pdf example on the Dice: best 0.80 -> 0, worst 0.60 -> 1, 0.78 -> 0.10
    return pd.DataFrame(
        {
            "team": ["A", "B", "C"],
            "dice": [0.80, 0.60, 0.78],
            "hd95_mm": [4.0, 8.0, 4.4],
            "avd_pct": [10.0, 30.0, 12.0],
            "lesion_recall": [0.8, 0.6, 0.78],
            "lesion_f1": [0.8, 0.6, 0.78],
        }
    )


def test_ranking_follows_the_readme_example():
    s = ranking(teams())
    # every metric places A at 0, B at 1 and C at 0.10: overall 0, 1 and 0.10
    assert s.tolist() == pytest.approx([0.0, 1.0, 0.10])


def test_position_inserts_one_method_among_the_teams():
    p = position(
        teams(),
        {"dice": 0.79, "hd95_mm": 4.2, "avd_pct": 11.0, "lesion_recall": 0.79, "lesion_f1": 0.79},
        "ours",
    )
    assert p["position"] == 2 and p["n_teams"] == 4
    assert p["rank_dice"] == 2 and p["rank_avd_pct"] == 2
