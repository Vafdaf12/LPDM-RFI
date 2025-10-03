#!/bin/bash

#SBATCH --job-name='LPDM_Testing'
#SBATCH --time=09:00:00
#SBATCH --ntasks=1
#SBATCH --partition=GPU
#SBATCH --constraint=p100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=20GB
#SBATCH --output=stdout.log
#SBATCH --error=stderr.log
#SBATCH --mail-user=u22563777@tuks.co.za
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80

echo "Loading anaconda"
module load anaconda3

echo "Activating conda environment"
conda activate lpdm

echo "Running testing"
cd scripts

# Run denoising
python denoise_config.py --base_path ../configs/test/denoise_rfi.yaml  --device cuda
