#!/bin/bash
#SBATCH --account=butlerry-deepctseg
#SBATCH --job-name=ct_mocov2_r50_train
#SBATCH --output=ct_mocov2_r50_train_%j.log
#SBATCH --error=ct_mocov2_r50_train_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=48:00:00
#SBATCH --qos=bbgpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pxg491@alumni.bham.ac.uk

# ================== Paths ==================
# Update this path to point to your new ResNet50 clean script
PYTHON_SCRIPT="/rds/projects/b/butlerry-deepctseg/40K_pipeline/part1/mocoV2_train/2026_5_1_mocoV2_train_r50.py"
H5_DATA_DIR="/rds/projects/b/butlerry-deepctseg/40K_pipeline/data_prep_results/split_h5_files/train/train_data.h5"

OUTPUT_DIR="/rds/projects/b/butlerry-deepctseg/40K_pipeline/part1/mocoV2_train/ct_mocoV2_r50_results"

# ================== Setup ==================
module purge
module load bluebear
module load bear-apps/2023a
module load Python/3.11.3-GCCcore-12.3.0
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
module load torchvision/0.16.0-foss-2023a-CUDA-12.1.1
module load h5py/3.9.0-foss-2023a
module load matplotlib/3.7.2-gfbf-2023a
module load OpenCV/4.8.1-foss-2023a-CUDA-12.1.1-contrib

# ================== Execution ==================
mkdir -p "${OUTPUT_DIR}"

echo "Starting ResNet50 MoCo V2 Training at $(date)"
echo "Targeting H5: ${H5_DATA_DIR}"

# Execution parameters:
# 1. batch_size 128: Optimized for ResNet50 memory footprint on A100.
# 2. mlp: Enables the projection head required for MoCo V2.
# 3. moco_k 8192: Maintains the queue size from your previous optimized run.

python3 "${PYTHON_SCRIPT}" \
    --h5_file "${H5_DATA_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_workers 10 \
    --batch_size 128 \
    --lr 0.015 \
    --moco_k 8192 \
    --epochs 600 \
    --mlp

if [ $? -eq 0 ]; then
    echo "## ResNet50 Training Finished Successfully."
else
    echo "## Training Failed. Check logs: ct_mocov2_r50_train_${SLURM_JOB_ID}.err"
fi