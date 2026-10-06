"""HD-BET worker: brain masks for a batch of T1 images, on the local GPU.

This file is executed **by path** in a separate, temporary uv environment that provides a CUDA build
of PyTorch and HD-BET (the project's own environment has a CPU-only PyTorch). It must therefore only
import the standard library, torch and HD_BET, never ``wmh_multisite``. It is launched by
``preproc/brainmask.py``, which runs it under the memory watchdog and checks every mask it writes.

Usage (normally not called by hand):
    python hdbet_worker.py jobs.json
with jobs.json = {"pairs": [[input_t1, output_mask], ...], "vram_cap_gb": 5.5, "tta": true}
"""

import json
import sys
import time

import torch


def main(job_file: str) -> None:
    job = json.load(open(job_file))
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    # Hard cap: above it PyTorch raises a clean out-of-memory error instead of letting the Windows
    # driver spill GPU memory into shared system RAM (the cause of the first local crash, J-005).
    torch.cuda.set_per_process_memory_fraction(min(job["vram_cap_gb"] / total_gb, 0.95), 0)

    from HD_BET.checkpoint_download import maybe_download_parameters
    from HD_BET.hd_bet_prediction import get_hdbet_predictor

    maybe_download_parameters()  # weights are cached after the first call
    predictor = get_hdbet_predictor(use_tta=job["tta"], device=torch.device("cuda"), verbose=False)
    t0 = time.time()
    # One preprocessing and one export process: HD-BET's defaults (4 and 8) are what raised the RAM
    # to 10.5 GB on Kaggle; with 1 and 1 it peaks at ~3.2 GB (J-033).
    predictor.predict_from_files(
        [[src] for src, _ in job["pairs"]],
        [dst for _, dst in job["pairs"]],
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=1,
        num_processes_segmentation_export=1,
        folder_with_segs_from_prev_stage=None,
        num_parts=1,
        part_id=0,
    )
    print(
        json.dumps(
            {
                "n": len(job["pairs"]),
                "seconds": round(time.time() - t0, 1),
                "torch_peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                "torch_peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 2),
            }
        )
    )


if __name__ == "__main__":  # required on Windows: nnU-Net starts worker processes with "spawn"
    main(sys.argv[1])
