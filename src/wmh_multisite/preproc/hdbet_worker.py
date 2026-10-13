"""HD-BET worker: brain masks for a batch of T1 images, on the local GPU.

Run by path from ``preproc/brainmask.py`` in a temporary uv environment with CUDA PyTorch and HD-BET,
so it may only import the standard library, torch and HD_BET, never ``wmh_multisite``.

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
    # Hard cap: PyTorch then raises a clean OOM instead of letting the Windows driver spill GPU memory
    # into system RAM, which crashed the machine once.
    torch.cuda.set_per_process_memory_fraction(min(job["vram_cap_gb"] / total_gb, 0.95), 0)

    from HD_BET.checkpoint_download import maybe_download_parameters
    from HD_BET.hd_bet_prediction import get_hdbet_predictor

    maybe_download_parameters()
    predictor = get_hdbet_predictor(use_tta=job["tta"], device=torch.device("cuda"), verbose=False)
    t0 = time.time()
    # One preprocessing and one export process: HD-BET's defaults (4 and 8) took 10.5 GB of RAM on
    # Kaggle, 1 and 1 peak at about 3.2 GB.
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
