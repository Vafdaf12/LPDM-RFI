#!/bin/bash

#SBATCH --job-name='LPDM_Training'
#SBATCH --time=09:00:00
#SBATCH --ntasks=1
#SBATCH --partition=GPU
#SBATCH --constraint=p100
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=20GB
#SBATCH --output=stdout.log
#SBATCH --error=stderr.log
#SBATCH --mail-user=u22563777@tuks.co.za
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80

echo "Loading anaconda"
module load anaconda3
conda activate lpdm

echo "Running training"

# Resume from last log points
python main.py --base configs/train/lpdm_rfi.yaml --gpu 1 --resume logs/$(ls logs | sort -n | tail -n 1)


# Start training from scratch
#python main.py --base configs/train/lpdm_rfi.yaml




