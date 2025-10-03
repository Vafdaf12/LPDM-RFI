#!/bin/bash

#SBATCH --job-name='LPDM_Inpainting'
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=8GB
#SBATCH --output=stdout_inpaint.log
#SBATCH --error=stderr_inpaint.log
#SBATCH --mail-user=u22563777@tuks.co.za
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80


module load anaconda3
conda activate lpdm 

# Perform inpainting of real-world data
python /users/vafdaf12/projects/lpdm-rfi/scripts/baseline_to_image.py workspace/data-real -o workspace/data-real/inpaint-mean --method mean
python /users/vafdaf12/projects/lpdm-rfi/scripts/baseline_to_image.py workspace/data-real -o workspace/data-real/inpaint-zero --method zero

# Perform inpainting of simulated data
python /users/vafdaf12/projects/lpdm-rfi/scripts/baseline_to_image.py workspace/data-simulated -o workspace/data-simulated/inpaint-mean --method mean
python /users/vafdaf12/projects/lpdm-rfi/scripts/baseline_to_image.py workspace/data-simulated -o workspace/data-simulated/inpaint-zero --method zero
