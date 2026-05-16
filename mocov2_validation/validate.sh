#!/bin/bash
#SBATCH --account=butlerry-deepctseg
#SBATCH --job-name=ct_mocov2_val_r50
#SBATCH --output=ct_mocov2_val_r50_%j.log
#SBATCH --error=ct_mocov2_val_r50_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100_80:1
#SBATCH --time=24:00:00
#SBATCH --qos=bbgpu
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pxg491@alumni.bham.ac.uk

H5_DATA_FILE="/rds/projects/b/butlerry-deepctseg/40K_pipeline/data_prep_results/split_h5_files/val/val_data.h5"
MODEL_PATH="/rds/projects/b/butlerry-deepctseg/40K_pipeline/part1/mocoV2_train/ct_mocoV2_r50_results/srxtm_moco_r50_e360.pth"
OUTPUT_DIR="/rds/projects/b/butlerry-deepctseg/40K_pipeline/part1/mocoV2_val/moco_val_r50_output_r50"
PYTHON_SCRIPT="/rds/projects/b/butlerry-deepctseg/40K_pipeline/part1/mocoV2_val/2026_05_02_mocoV2_val_r50.py"

module purge
module load bluebear
module load bear-apps/2023a
module load Python/3.11.3-GCCcore-12.3.0
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
module load torchvision/0.16.0-foss-2023a-CUDA-12.1.1
module load h5py/3.9.0-foss-2023a
module load scikit-image/0.22.0-foss-2023a
module load umap-learn/0.5.5-foss-2023a

export OMP_NUM_THREADS=10
export MKL_NUM_THREADS=10

python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"

mkdir -p "${OUTPUT_DIR}"

python3 "${PYTHON_SCRIPT}" \
    --h5_file "${H5_DATA_FILE}" \
    --model_path "${MODEL_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size 64

echo "ResNet50 Validation Job finished."