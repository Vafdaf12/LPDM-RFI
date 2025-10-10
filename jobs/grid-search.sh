#!/bin/bash

#SBATCH --job-name='lpdm-grid-search'
#SBATCH --time=09:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=20GB
#SBATCH --output=logs_slurm/%A_%a_out.log
#SBATCH --error=logs_slurm/%A_%a_err.log
#SBATCH --partition=GPU
#SBATCH --constraint=p100
#SBATCH --gres=gpu:1
#SBATCH --mail-user=u22563777@tuks.co.za
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80
#SBATCH --array=0-2 # job array index

METHODS=(noise mean zero)
METHOD=${METHODS[SLURM_ARRAY_TASK_ID]}

echo "Loading anaconda"
module load anaconda3

echo "Activating conda environment"
conda activate lpdm

echo "Running testing"
cd scripts

for config in $(find ../workspace/grid-search/configs/$METHOD -type f); do
    echo "---------------------------------------------------------------- $method  $config"
    python denoise_config.py $config  --device cuda --overwrite
done
