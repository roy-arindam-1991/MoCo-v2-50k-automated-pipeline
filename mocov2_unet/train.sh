#!/bin/bash
#SBATCH --account=butlerry-deepctseg
#SBATCH --job-name=ct_mocov2_r50_unet_train
#SBATCH --output=ct_mocov2_r50_unet_%j.log
#SBATCH --error=ct_mocov2_r50_unet_%j.err
#SBATCH --time=48:00:00                
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G                     
#SBATCH --gres=gpu:a100:1              
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pxg491@alumni.bham.ac.uk
#SBATCH --qos=bbgpu

set -e 

# Performance tuning for HPC cluster 
export OMP_NUM_THREADS=16
export HDF5_USE_FILE_LOCKING=FALSE
export TORCH_NUM_THREADS=16

module purge
module load bluebear
module load bear-apps/2023a
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load torchvision/0.16.0-foss-2023a-CUDA-12.1.1
module load h5py/3.9.0-foss-2023a
module load tqdm/4.66.1-GCCcore-12.3.0
module load matplotlib/3.7.2-gfbf-2023a

# Paths updated for ResNet50 backbone results
SCRIPT_PATH="/rds/projects/b/butlerry-deepctseg/40K_pipeline/part1/mocoV2_unet/2026_05_02_mocov2_unet_r50.py"
IMAGE_ROOT="/rds/projects/b/butlerry-deepctseg/40K_pipeline/data_prep_results/40k_images_fixed.h5"
MASK_ROOT="/rds/projects/b/butlerry-deepctseg/40K_pipeline/data_masking/results/test_masks_h5/ct_masks_fixed.h5"
BACKBONE_PATH="/rds/projects/b/butlerry-deepctseg/40K_pipeline/part1/mocoV2_train/ct_mocoV2_r50_results/srxtm_moco_r50_e360.pth"
OUTPUT_DIR="/rds/projects/b/butlerry-deepctseg/40K_pipeline/part1/mocoV2_unet/mocov2_r50_unet_results"

mkdir -p "${OUTPUT_DIR}"

python "$SCRIPT_PATH" \
    --image_root "$IMAGE_ROOT" \
    --mask_root "$MASK_ROOT" \
    --backbone_path "$BACKBONE_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --epochs 500 \
    --batch_size 16 \
    --lr 0.0001 \
    --win_min -500 \
    --win_max 1300 \
    --workers 16 \
    --resume