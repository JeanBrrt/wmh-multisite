#!/bin/bash
# nnU-Net training of the 5 cross-validation folds on Jean Zay, one GPU per fold (ROADMAP 9.4, J-062).
#
# NOT TESTED (no Jean Zay account): written from the IDRIS documentation; the account, the partition
# constraint and the module names are to check before the first run. On Kaggle, the project trained
# fold 0 only (budget, J-037); this job array trains the 5 folds in parallel, for the ensemble that
# the leaderboard teams use.
#
# Before:
#   1. Build containers/apptainer.def elsewhere, copy the .sif, `idrcontmgr cp wmh-multisite.sif`.
#   2. Copy data/nnunet/nnUNet_raw/Dataset001_WMH (export of step 5.1) to $WORK/wmh/data/nnunet/.
#   3. Preprocess once (CPU partition, or: sbatch --dependency on a short job running)
#        nnUNetv2_plan_and_preprocess -d 1 -pl nnUNetPlannerResEncM -c 3d_fullres
#      then copy splits_final.json (folds stratified by scanner and lesion load, J-032) into
#      nnUNet_preprocessed/Dataset001_WMH/.
# Run:   sbatch hpc/slurm_jeanzay.sh            (5 jobs: folds 0 to 4)
# Resume a fold stopped by the time limit: resubmit; `--c` restarts from the last checkpoint.

#SBATCH --job-name=wmh-nnunet
#SBATCH --account=CHANGEME@v100          # project allocation (V100 hours)
#SBATCH --constraint=v100-32g            # 32 GB V100
#SBATCH --qos=qos_gpu-t3                 # up to 20 h per job
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10               # data augmentation workers (J-019: the T4 run was GPU-bound)
#SBATCH --hint=nomultithread
#SBATCH --array=0-4                      # one job per fold
#SBATCH --output=logs/%x_fold%a_%j.out
#SBATCH --error=logs/%x_fold%a_%j.err

set -euo pipefail

FOLD=${SLURM_ARRAY_TASK_ID}
TRAINER=${TRAINER:-nnUNetTrainerWMH_250}          # or nnUNetTrainerWMH_DA5_250 (J-052)
DATA=${DATA:-$WORK/wmh/data}
IMAGE=${IMAGE:-$SINGULARITY_ALLOWED_DIR/wmh-multisite.sif}

module purge
module load singularity

mkdir -p logs
echo "fold ${FOLD} | trainer ${TRAINER} | node $(hostname) | $(date)"

RESUME=""
if ls "${DATA}/nnunet/nnUNet_results/Dataset001_WMH/${TRAINER}__nnUNetResEncUNetMPlans__3d_fullres/fold_${FOLD}/checkpoint_latest.pth" >/dev/null 2>&1; then
    RESUME="--c"
fi

singularity exec --nv \
    --bind "${DATA}:/opt/wmh-multisite/data" \
    "${IMAGE}" \
    nnUNetv2_train 1 3d_fullres "${FOLD}" -tr "${TRAINER}" -p nnUNetResEncUNetMPlans --npz ${RESUME}

echo "done fold ${FOLD} | $(date)"
# After the 5 folds: ensemble prediction on the test set (GPU job, ~1 h):
#   nnUNetv2_predict -i <imagesTs> -o <out> -d 1 -c 3d_fullres -tr ${TRAINER} -p nnUNetResEncUNetMPlans -f 0 1 2 3 4
