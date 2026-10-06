"""SyN settings benchmark (step 3.6): subjects x variants, interleaved subject by subject.

Usage (from the repository root):
    uv run python scripts/syn_benchmark.py         # CC only, notebook-06 subjects, sub-021 first
    uv run python scripts/syn_benchmark.py --variants default_mattes_40-20-0 mattes_100-70-50-20 \
        --subjects sub-049 sub-021 --jobs 4 --threads 1         # the first run (deterministic, 1 thread)

Speed: by default ONE task at a time with ALL the CPU threads (ITK multithreading), so that the first
subject is finished as early as possible. Several ITK threads make the result very slightly
non-deterministic (J-031): fine for a benchmark, not for the pipeline.

Progress: START / END lines, and every minute a heartbeat with the elapsed time, the ANTs stage, the
resolution level and the iteration reached (parsed from the ANTs verbose output, kept in
``<task folder>/ants.log``). Everything is also written to
data/derivatives/wmh-multisite/_syn_benchmark/bench.log. Results are appended as each task ends to
results/tables/syn_benchmark.csv. Re-running skips the tasks already done.

Each task = one subject with one variant: SyN registration (brain to brain) + atlas ventricles carried
onto the FLAIR grid + refinement, all written to a scratch folder (the pipeline derivatives are not
touched).
"""

import argparse
import csv
import itertools
import logging
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "derivatives" / "wmh-multisite" / "_syn_benchmark"  # transforms, masks, log
NOTEBOOK_SUBJECTS = ["sub-021", "sub-000", "sub-002", "sub-049", "sub-053"]  # notebook 06, sub-021 first
VARIANTS = {
    "default_mattes_40-20-0": {"reg_iterations": (40, 20, 0)},
    "mattes_100-70-50-20": {"reg_iterations": (100, 70, 50, 20)},
    "cc_100-70-50-20": {"reg_iterations": (100, 70, 50, 20), "syn_metric": "CC", "syn_sampling": 4},
    # same on the 2 mm template (8x fewer voxels); CC radius in voxels halved to keep ~4 mm (7.5)
    "cc2mm_100-70-50-20": {
        "reg_iterations": (100, 70, 50, 20),
        "syn_metric": "CC",
        "syn_sampling": 2,
        "template_resolution": 2,
    },
}
FIELDS = [
    "subject",
    "variant",
    "status",
    "syn_seconds",
    "peak_ram_gb",
    "syn_corr_brain",
    "tpl_vent_on_csf",
    "vent_atlas_ml",
    "vent_ml",
    "dice_atlas_refined",
    "error",
]


def redirect_stdout(path: Path) -> None:
    """Send file descriptor 1 and sys.stdout to ``path`` for the rest of this (child) process.

    ANTs prints its verbose output twice: from C++ on file descriptor 1, and from Python (print) on
    sys.stdout. In a child process spawned from a Git Bash console both can be bound to an invalid
    handle ("WinError 6"). Nothing is restored: the child ends with the task.
    """
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    os.dup2(fd, 1)
    os.close(fd)
    sys.stdout = open(1, "w", buffering=1, closefd=False, errors="replace")


def task(sub: str, variant: str) -> dict:
    """Runs in a guarded child process (ITK threads inherited from the parent)."""
    import shutil

    import ants
    import numpy as np

    from wmh_multisite.config import load_config
    from wmh_multisite.preproc import atlas

    cfg = load_config()
    spec = dict(VARIANTS[variant])
    resolution = spec.pop("template_resolution", cfg["preproc"]["atlas"]["resolution"])
    cfg["preproc"]["atlas"]["resolution"] = resolution
    tpl = atlas.template_files(cfg)
    p = atlas.subject_paths(cfg, sub)
    work = OUT / f"{sub}__{variant}"
    work.mkdir(parents=True, exist_ok=True)
    for k in ("affine", "warp", "invwarp", "vent_t1", "vent_atlas", "vent"):  # redirect every output
        p[k] = work / p[k].name
    fixed = atlas.masked(
        ants.image_read(str(tpl["t1"]), pixeltype="float"), ants.image_read(str(tpl["brain"]))
    )
    moving = atlas.masked(
        ants.image_read(str(p["t1_n4"]), pixeltype="float"), ants.image_read(str(p["t1_brain"]))
    )
    redirect_stdout(work / "ants.log")
    ants.config._random_seed = 42
    t0 = time.time()
    reg = ants.registration(fixed=fixed, moving=moving, type_of_transform="SyN", verbose=True, **spec)
    seconds = time.time() - t0
    shutil.copyfile(reg["fwdtransforms"][1], p["affine"])
    shutil.copyfile(reg["fwdtransforms"][0], p["warp"])
    shutil.copyfile(reg["invtransforms"][1], p["invwarp"])
    from wmh_multisite.preproc.register import remove_registration_files

    remove_registration_files(reg)
    brain = ants.image_read(str(tpl["brain"])).numpy() > 0.5
    warped, f = reg["warpedmovout"].numpy(), fixed.numpy()
    labels = ants.image_read(str(tpl["labels"])).numpy()
    tpl_vent = np.isin(labels, cfg["preproc"]["atlas"]["ventricle_labels"])
    dark = warped < np.percentile(warped[brain], 8)  # CSF of the warped subject
    atlas.ventricles_to_flair(cfg, p, tpl)
    stats = atlas.refine_ventricles(cfg, p)
    return {
        "syn_seconds": round(seconds, 1),
        "syn_corr_brain": round(float(np.corrcoef(warped[brain], f[brain])[0, 1]), 4),
        "tpl_vent_on_csf": round(float(dark[tpl_vent].mean()), 3),
        "vent_atlas_ml": stats["vent_atlas_ml"],
        "vent_ml": stats["vent_ml"],
        "dice_atlas_refined": stats["dice_atlas_refined"],
    }


def ants_progress(log_file: Path) -> str:
    """Last ANTs stage, level and iteration found in the verbose log (best effort)."""
    # Format (antsRegistration -v 1): "Stage 0" (affine) then "Stage 1" (SyN), each with
    # "iterations = 100x70x50x20"; one "DIAGNOSTIC,Iteration,..." header per resolution level, then one
    # " 1DIAGNOSTIC,    12, <metric>, ..." line per iteration.
    try:
        text = log_file.read_text(errors="ignore")
    except OSError:
        return "no ANTs output yet"
    stages = list(re.finditer(r"^Stage (\d+)\s*$", text, re.MULTILINE))
    if not stages:
        return "initializing"
    chunk = text[stages[-1].end() :]
    kind = re.findall(r"\*\*\* Running (\w+) registration", chunk)
    planned = re.findall(r"iterations = ([\dx]+)", chunk)
    planned = [int(n) for n in planned[0].split("x")] if planned else []
    level = len(re.findall(r"DIAGNOSTIC,Iteration", chunk))
    it = re.findall(r"\dDIAGNOSTIC,\s*(\d+),", chunk)
    out = f"stage {stages[-1].group(1)} ({kind[-1] if kind else '?'})"
    if level:
        out += f", level {level}/{len(planned) or '?'}"
        if it and planned and level <= len(planned):
            out += f", iteration {it[-1]}/{planned[level - 1]}"
    if "Elapsed time (stage" in chunk:
        out += ", stage done"
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="SyN settings benchmark (step 3.6).")
    parser.add_argument("--variants", nargs="+", default=["cc_100-70-50-20"], choices=list(VARIANTS))
    parser.add_argument("--subjects", nargs="+", default=NOTEBOOK_SUBJECTS, help='subject ids, or "all"')
    parser.add_argument("--jobs", type=int, default=1, help="tasks in parallel")
    parser.add_argument("--threads", type=int, default=os.cpu_count(), help="ITK threads per task")
    parser.add_argument("--heartbeat", type=float, default=60, help="seconds between progress lines")
    args = parser.parse_args()

    import pandas as pd

    from wmh_multisite.config import load_config, resolve_path
    from wmh_multisite.preproc.atlas import template_files
    from wmh_multisite.utils.guard import call_guarded, wait_for_free_ram

    OUT.mkdir(exist_ok=True)
    log = logging.getLogger("syn_bench")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(OUT / "bench.log"), logging.StreamHandler(sys.stdout)],
    )
    cfg = load_config()
    template_files(cfg)
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(args.threads)
    ram_limit = cfg["guard"]["ram_limit_gb"]
    csv_path = ROOT / "results" / "tables" / "syn_benchmark.csv"
    done = set()
    if csv_path.is_file():
        done = {(r["subject"], r["variant"]) for r in csv.DictReader(open(csv_path)) if r["status"] == "ok"}
    subjects = args.subjects
    if subjects == ["all"]:  # every subject, scanners interleaved: a run stopped early covers every site
        part = pd.read_csv(resolve_path(cfg, "bids") / "participants.tsv", sep="	")
        by_scanner = [g.participant_id.tolist() for _, g in part.groupby("scanner")]
        subjects = [s for group in itertools.zip_longest(*by_scanner) for s in group if s]
    tasks = [(s, v) for s in subjects for v in args.variants if (s, v) not in done]
    log.info(
        "SyN benchmark: %d tasks (%d already done), %d in parallel, %d ITK threads each, RAM limit %s GB,"
        " order: %s",
        len(tasks),
        len(done),
        args.jobs,
        args.threads,
        ram_limit,
        [f"{s}/{v.split('_')[0]}" for s, v in tasks],
    )

    running: dict[tuple[str, str], float] = {}
    stop = threading.Event()

    def heartbeat() -> None:
        while not stop.wait(args.heartbeat):
            for (sub, variant), t0 in list(running.items()):
                log.info(
                    "...   %s | %s | %.1f min | %s",
                    sub,
                    variant,
                    (time.time() - t0) / 60,
                    ants_progress(OUT / f"{sub}__{variant}" / "ants.log"),
                )

    def run_one(sub: str, variant: str) -> dict:
        wait_for_free_ram(
            cfg["guard"]["min_free_ram_gb"], log=lambda m: log.info("%s %s %s", sub, variant, m)
        )
        log.info("START %s | %s", sub, variant)
        running[(sub, variant)] = time.time()
        try:
            r = call_guarded(task, sub, variant, ram_limit_gb=ram_limit)
        finally:
            running.pop((sub, variant), None)
        row = {
            "subject": sub,
            "variant": variant,
            "status": "ok" if r.ok else "failed",
            "peak_ram_gb": r.peak_ram_gb,
            "error": "" if r.ok else str(r.killed or r.error)[-200:],
        }
        if r.ok:
            row.update(r.value)
        return row

    threading.Thread(target=heartbeat, daemon=True).start()
    new_file = not csv_path.is_file()
    with ThreadPoolExecutor(max_workers=args.jobs) as pool, open(csv_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        futures = [pool.submit(run_one, s, v) for s, v in tasks]
        for fut in as_completed(futures):
            row = fut.result()
            writer.writerow(row)
            fh.flush()
            log.info(
                "END   %s | %-24s | %s | %s s | RAM %s GB | corr %s | tpl vent on CSF %s"
                " | vent atlas %s ml -> refined %s ml %s",
                row["subject"],
                row["variant"],
                row["status"],
                row.get("syn_seconds"),
                row["peak_ram_gb"],
                row.get("syn_corr_brain"),
                row.get("tpl_vent_on_csf"),
                row.get("vent_atlas_ml"),
                row.get("vent_ml"),
                row["error"],
            )
    stop.set()
    log.info("benchmark finished")


if __name__ == "__main__":
    main()
