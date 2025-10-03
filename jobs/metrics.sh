#!/bin/bash

#SBATCH --job-name='LPDM_Metrics'
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=8GB
#SBATCH --output=stdout_metrics.log
#SBATCH --error=stderr_metrics.log
#SBATCH --mail-user=u22563777@tuks.co.za
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80


module load anaconda3
conda activate lpdm 

# Calculate metrics for real-world data
python /users/vafdaf12/projects/lpdm-rfi/scripts/calculate_metrics.py \
    --target datasets/rfi/sim-test-target.npy \
    --pred tmp/phi200_s10.npy

# python /users/vafdaf12/projects/lpdm-rfi/scripts/calculate_metrics.py \
#     --target workspace/data-real/inpaint-mean/target \
#     --pred workspace/data-real/inpaint-mean/eta \
#     --output workspace/data-real/inpaint-mean/metrics_eta.csv
# 
# python /users/vafdaf12/projects/lpdm-rfi/scripts/calculate_metrics.py \
#     --target workspace/data-real/inpaint-zero/target \
#     --pred workspace/data-real/inpaint-zero/pred/lpdm_rfi/phi200_s10 \
#     --output workspace/data-real/inpaint-zero/metrics_pred.csv
# 
# python /users/vafdaf12/projects/lpdm-rfi/scripts/calculate_metrics.py \
#     --target workspace/data-real/inpaint-zero/target \
#     --pred workspace/data-real/inpaint-zero/eta \
#     --output workspace/data-real/inpaint-zero/metrics_eta.csv

# Calculate metrics for simulated data
# python /users/vafdaf12/projects/lpdm-rfi/scripts/calculate_metrics.py \
#     --target workspace/data-simulated/inpaint-mean/target \
#     --pred workspace/data-simulated/inpaint-mean/pred/lpdm_rfi/phi200_s10 \
#     --output workspace/data-simulated/inpaint-mean/metrics_pred.csv
# 
# python /users/vafdaf12/projects/lpdm-rfi/scripts/calculate_metrics.py \
#     --target workspace/data-simulated/inpaint-mean/target \
#     --pred workspace/data-simulated/inpaint-mean/eta \
#     --output workspace/data-simulated/inpaint-mean/metrics_eta.csv
# 
# python /users/vafdaf12/projects/lpdm-rfi/scripts/calculate_metrics.py \
#     --target workspace/data-simulated/inpaint-zero/target \
#     --pred workspace/data-simulated/inpaint-zero/pred/lpdm_rfi/phi200_s10 \
#     --output workspace/data-simulated/inpaint-zero/metrics_pred.csv
# 
# python /users/vafdaf12/projects/lpdm-rfi/scripts/calculate_metrics.py \
#     --target workspace/data-simulated/inpaint-zero/target \
#     --pred workspace/data-simulated/inpaint-zero/eta \
#     --output workspace/data-simulated/inpaint-zero/metrics_eta.csv
