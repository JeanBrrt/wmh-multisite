"""nnU-Net inference worker, run on the local GPU in a temporary uv environment (CUDA PyTorch + nnU-Net).

Executed **by path** (not imported): it only imports the standard library, torch and nnunetv2, never
``wmh_multisite``. Launched by ``seg/nnunet_infer.py`` under the memory watchdog.

The model was trained with a custom trainer (``nnUNetTrainerWMH_250``, defined on Kaggle only: same
network as ``nnUNetTrainer``, only the number of epochs and the checkpoint period differ). It is not
installed locally, so the network is rebuilt from ``plans.json`` with the standard trainer's
``build_network_architecture`` and the weights of ``checkpoint_final.pth`` are loaded into it
(``nnUNetPredictor.manual_initialization``) instead of ``initialize_from_trained_model_folder``.

Usage (normally not called by hand):
    python nnunet_infer_worker.py job.json
with job.json = {"model_dir": ..., "fold": 0, "checkpoint": "checkpoint_final.pth",
                 "cases": [[flair, t1, output_without_ending], ...], "vram_cap_gb": 5.5,
                 "use_mirroring": true, "tile_step_size": 0.5}
"""

import json
import os
import sys
import time

import torch


def main(job_file: str) -> None:
    job = json.load(open(job_file))
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    # Hard cap: a clean out-of-memory error instead of a spill into Windows shared memory (J-005).
    torch.cuda.set_per_process_memory_fraction(min(job["vram_cap_gb"] / total_gb, 0.95), 0)

    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    model_dir = job["model_dir"]
    dataset_json = json.load(open(os.path.join(model_dir, "dataset.json")))
    dataset_json["file_ending"] = ".nii.gz"  # Kaggle trained on decompressed .nii; we read and write .nii.gz
    plans_manager = PlansManager(os.path.join(model_dir, "plans.json"))
    checkpoint = torch.load(
        os.path.join(model_dir, f"fold_{job['fold']}", job["checkpoint"]),
        map_location="cpu",
        weights_only=False,
    )  # our own checkpoint: trusted
    configuration_manager = plans_manager.get_configuration(checkpoint["init_args"]["configuration"])
    n_in = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
    n_out = plans_manager.get_label_manager(dataset_json).num_segmentation_heads
    network = nnUNetTrainer.build_network_architecture(
        plans_manager, configuration_manager, n_in, n_out, enable_deep_supervision=False
    )

    predictor = nnUNetPredictor(
        tile_step_size=job["tile_step_size"],
        use_gaussian=True,
        use_mirroring=job["use_mirroring"],
        perform_everything_on_device=True,
        device=torch.device("cuda"),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    predictor.manual_initialization(
        network,
        plans_manager,
        configuration_manager,
        [checkpoint["network_weights"]],
        dataset_json,
        checkpoint["trainer_name"],
        checkpoint.get("inference_allowed_mirroring_axes"),
    )
    t0 = time.time()
    # 1 preprocessing and 1 export process keep the RAM low (same reasoning as HD-BET, J-033).
    predictor.predict_from_files(
        [[flair, t1] for flair, t1, _ in job["cases"]],
        [out for _, _, out in job["cases"]],
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=1,
        num_processes_segmentation_export=1,
    )
    print(
        json.dumps(
            {
                "n": len(job["cases"]),
                "seconds": round(time.time() - t0, 1),
                "trainer_name": checkpoint["trainer_name"],
                "epoch": checkpoint.get("current_epoch"),
                "torch_peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                "torch_peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 2),
            }
        )
    )


if __name__ == "__main__":  # required on Windows: nnU-Net starts worker processes with "spawn"
    main(sys.argv[1])
