"""Carbon footprint of the project (ROADMAP step 9.5, J-062).

Two kinds of numbers, never mixed up (column ``method``):

1. **measured**: CodeCarbon (offline tracker, France grid) around stages that can be re-run locally
   without changing any output (computed in memory, nothing written): official evaluation of M1 on the
   110 test cases, NAWM features of the 170 subjects (8.2), site classifier and ComBat (8.3).
2. **estimated**: the long runs already done (Kaggle trainings, WMH-SynthSeg, SyN, HD-BET, inference)
   cannot be measured afterwards: energy = power x duration x PUE, emissions = energy x carbon
   intensity. Durations come from the decision log; powers are hardware TDPs (upper bounds); the
   Kaggle region is unknown, so the world average intensity is used. Orders of magnitude only.

Output: results/tables/carbon_footprint.csv

Usage:
    uv run python scripts/carbon_footprint.py
"""

from __future__ import annotations

import logging
import sys
import tempfile
import time

import pandas as pd

from wmh_multisite.config import load_config, resolve_path

log = logging.getLogger(__name__)

WORLD_KG_PER_KWH = 0.475  # world average carbon intensity used by CodeCarbon when the region is unknown
CLOUD_PUE = 1.1  # data centre overhead (Google reports ~1.1)

# stage, where, hardware, duration (h), power (W), PUE, intensity (kg/kWh, None = local measured), source
ESTIMATED = [
    (
        "nnU-Net M1 training (250 epochs)",
        "Kaggle",
        "2 x T4 (70 W) + CPU (40 W)",
        10.55,
        180,
        CLOUD_PUE,
        WORLD_KG_PER_KWH,
        "J-038",
    ),
    (
        "nnU-Net DA5 training (250 epochs)",
        "Kaggle",
        "2 x T4 (70 W) + CPU (40 W)",
        250 * 143 / 3600,
        180,
        CLOUD_PUE,
        WORLD_KG_PER_KWH,
        "J-052",
    ),
    (
        "WMH-SynthSeg, 170 subjects (CPU)",
        "Kaggle",
        "4 vCPU, 30 GB RAM (40 W)",
        170 * 350 / 3600,
        40,
        CLOUD_PUE,
        WORLD_KG_PER_KWH,
        "J-053",
    ),
    ("SyN to MNI, 170 subjects", "local", "laptop CPU, 12 threads (60 W)", 8.0, 60, 1.0, None, "J-054"),
    (
        "HD-BET brain masks, 170 subjects",
        "local",
        "RTX 4060 laptop + CPU (100 W)",
        47 / 60,
        100,
        1.0,
        None,
        "J-033",
    ),
    (
        "nnU-Net inference, 3 x 110 cases",
        "local",
        "RTX 4060 laptop + CPU (100 W)",
        3 * 18.5 / 60,
        100,
        1.0,
        None,
        "J-052",
    ),
    (
        "nnU-Net robustness, 160 inferences",
        "local",
        "RTX 4060 laptop + CPU (100 W)",
        1769 / 3600,
        100,
        1.0,
        None,
        "J-058",
    ),
]


def measured_stages(cfg: dict) -> list[tuple[str, callable]]:
    from wmh_multisite.evaluate.metrics import evaluate
    from wmh_multisite.harmonize import combat, features

    part = pd.read_csv(resolve_path(cfg, "bids") / "participants.tsv", sep="\t")
    test = part[part.split == "test"].participant_id.tolist()
    return [
        ("official evaluation of M1, 110 test cases", lambda: evaluate(cfg, test, "resencm", jobs=4)),
        ("NAWM features, 170 subjects (8.2)", lambda: features.build(cfg, jobs=4)),
        ("site classifier + ComBat (8.3)", lambda: combat.run(cfg)),
    ]


def measure(cfg: dict) -> tuple[list[dict], float]:
    from codecarbon import OfflineEmissionsTracker

    rows, intensities = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for name, run in measured_stages(cfg):
            tracker = OfflineEmissionsTracker(
                country_iso_code="FRA",
                output_dir=tmp,
                log_level="error",
                save_to_file=False,
                project_name=name,
            )
            tracker.start()
            t0 = time.time()
            run()
            kg = tracker.stop()
            kwh = tracker.final_emissions_data.energy_consumed
            hours = (time.time() - t0) / 3600
            intensities.append(kg / kwh if kwh else float("nan"))
            rows.append(
                {
                    "stage": name,
                    "where": "local",
                    "hardware": "laptop (CodeCarbon)",
                    "hours": hours,
                    "power_w": kwh * 1000 / hours if hours else float("nan"),
                    "pue": 1.0,
                    "kwh": kwh,
                    "kg_co2eq": kg,
                    "method": "measured",
                    "source": "CodeCarbon",
                }
            )
            log.info("%s: %.1f min, %.4f kWh, %.4f kg CO2eq", name, hours * 60, kwh, kg)
    return rows, float(pd.Series(intensities).median())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    cfg = load_config()
    rows, local_intensity = measure(cfg)
    log.info("local carbon intensity used by CodeCarbon (France): %.3f kg CO2eq/kWh", local_intensity)
    for stage, where, hardware, hours, power, pue, intensity, source in ESTIMATED:
        kwh = power * hours * pue / 1000
        rows.append(
            {
                "stage": stage,
                "where": where,
                "hardware": hardware,
                "hours": hours,
                "power_w": power,
                "pue": pue,
                "kwh": kwh,
                "kg_co2eq": kwh * (intensity or local_intensity),
                "method": "estimated",
                "source": source,
            }
        )
    table = pd.DataFrame(rows)
    out = resolve_path(cfg, "results") / "tables" / "carbon_footprint.csv"
    table.to_csv(out, index=False, float_format="%.4g")
    pd.set_option("display.width", 200)
    log.info(
        "\n%s",
        table[["stage", "where", "hours", "kwh", "kg_co2eq", "method"]].round(3).to_string(index=False),
    )
    log.info(
        "total: %.1f kWh, %.2f kg CO2eq (Kaggle %.2f, local %.2f)",
        table.kwh.sum(),
        table.kg_co2eq.sum(),
        table[table["where"] == "Kaggle"].kg_co2eq.sum(),
        table[table["where"] == "local"].kg_co2eq.sum(),
    )


if __name__ == "__main__":
    main()
