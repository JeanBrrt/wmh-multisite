"""QC report: table of the IQM classes + thumbnails of the flagged images (ROADMAP step 4.4, J-051).

A single self-contained HTML file (thumbnails embedded as PNG): the tool of the visual control that 4.2
recommends. For every subject classed "check" or "exclude": central axial, coronal and sagittal slices
of the raw FLAIR and T1, with its worst metrics.

**Never published**: the thumbnails are patient images (CC BY-NC 4.0 data). The report is written to
``results/qc/``, which git ignores (``results/*``); only aggregated numbers go to the README.

Usage:
    uv run python -m wmh_multisite.viz.report
"""

from __future__ import annotations

import argparse
import base64
import html
import io as pyio
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from wmh_multisite.config import load_config, resolve_path
from wmh_multisite.qc.iqm import subject_paths
from wmh_multisite.utils import io

log = logging.getLogger(__name__)


def thumbnail(path: Path) -> str:
    """Base64 PNG of the three central slices of an image (canonical orientation, 1-99 % window)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import nibabel as nib

    img = nib.as_closest_canonical(io.load(path))
    data = img.get_fdata(dtype=np.float32)
    zooms = img.header.get_zooms()[:3]
    lo, hi = np.percentile(data[data > 0], [1, 99]) if (data > 0).any() else (0, 1)
    cx, cy, cz = (s // 2 for s in data.shape)
    views = [
        (data[:, :, cz].T, zooms[1] / zooms[0]),
        (data[:, cy, :].T, zooms[2] / zooms[0]),
        (data[cx].T, zooms[2] / zooms[1]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.6))
    for ax, (v, aspect) in zip(axes, views, strict=True):
        ax.imshow(v, cmap="gray", origin="lower", vmin=lo, vmax=hi, aspect=aspect)
        ax.axis("off")
    fig.tight_layout(pad=0.2)
    buf = pyio.BytesIO()
    fig.savefig(buf, format="png", dpi=70)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def build_html(flags: pd.DataFrame, thumbs: dict[str, dict[str, str]]) -> str:
    counts = pd.crosstab(flags.scanner, flags.qc_class)
    zcols = [c for c in flags.columns if c.startswith("z_")]
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'><title>QC report</title>",
        "<style>body{font-family:sans-serif;margin:16px;max-width:1100px}"
        "table{border-collapse:collapse;font-size:13px}"
        "td,th{border:1px solid #ccc;padding:3px 6px;text-align:right}"
        ".check{background:#fff3cd}.exclude{background:#f8d7da}img{max-width:100%}</style></head><body>",
        "<h1>wmh-multisite: image quality control</h1>",
        "<p>Local report, never published (patient images, CC BY-NC data). Classes: robust z per scanner "
        "(positive = worse) and Isolation Forest (step 4.2). "
        "A class is a recommendation for visual control.</p>",
        "<h2>Classes per scanner</h2>",
        counts.to_html(),
        "<h2>Flagged images</h2>",
    ]
    flagged = flags[flags.qc_class.isin(["check", "exclude"])].sort_values("max_z", ascending=False)
    if flagged.empty:
        parts.append("<p>No flagged image.</p>")
    for sub, row in flagged.iterrows():
        worst = row[zcols].astype(float).sort_values(ascending=False).head(3)
        worst_txt = ", ".join(f"{html.escape(k.removeprefix('z_'))} = {v:.1f}" for k, v in worst.items())
        parts.append(
            f"<h3 class='{row.qc_class}'>{html.escape(sub)} ({html.escape(str(row.scanner))}): "
            f"{row.qc_class}, max z = {row.max_z:.1f}</h3><p>Worst metrics: {worst_txt}</p>"
        )
        for modality, b64 in thumbs.get(sub, {}).items():
            parts.append(
                f"<p>{modality}</p>"
                f"<img alt='{html.escape(sub)} {modality}' src='data:image/png;base64,{b64}'>"
            )
    parts.append("<h2>All subjects</h2>")
    parts.append(flags[["scanner", "qc_class", "max_z", "worst_metric", "iforest_score"]].round(2).to_html())
    parts.append("</body></html>")
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Local HTML QC report (4.4).")
    parser.add_argument("--config")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    cfg = load_config(args.config)
    flags = pd.read_csv(resolve_path(cfg, "results") / "tables" / "qc_flags.csv", index_col="participant_id")
    thumbs = {}
    for sub in flags[flags.qc_class.isin(["check", "exclude"])].index:
        p = subject_paths(cfg, sub)
        thumbs[sub] = {m: thumbnail(p[m]) for m in ("FLAIR", "T1w")}
    out = resolve_path(cfg, "results") / "qc" / "qc_report.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_html(flags, thumbs), encoding="utf-8")
    log.info("%s (%d flagged subjects with thumbnails)", out, len(thumbs))


if __name__ == "__main__":
    main()
