"""Custom nnU-Net trainers of the project (J-037, J-052): 250 epochs, a checkpoint every 50 epochs.

nnU-Net finds trainers by class name inside its own package, so this file is copied into
``nnunetv2/training/nnUNetTrainer/variants/`` by the containers (``containers/``) and by the Kaggle
notebooks (``kaggle/03`` and ``04``, which write the same code). Single source for the HPC runs.
"""

import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerDA5 import nnUNetTrainerDA5


class nnUNetTrainerWMH_250(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),  # noqa: B008 (signature of nnU-Net trainers)
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250
        self.save_every = 50


class nnUNetTrainerWMH_DA5_250(nnUNetTrainerDA5):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),  # noqa: B008 (signature of nnU-Net trainers)
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250
        self.save_every = 50
