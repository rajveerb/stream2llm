#!/bin/bash
#SBATCH -J H200_crawler_full
#SBATCH --account=gts-krong6
#SBATCH -q inferno
#SBATCH -N 1
#SBATCH --mem=800GB
#SBATCH --ntasks-per-node=1
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:H200:2
#SBATCH -o H200_crawler_full_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rr@gatech.edu

CONDA_ENV=/storage/home/hcoda1/1/rbachkaniwala3/scratch/conda-envs/stream2llm
PROJECT_DIR=/storage/home/hcoda1/1/rbachkaniwala3/scratch/stream2llm

module load anaconda3
eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV

cd $PROJECT_DIR

# Crawler main experiment: all schedulers x 4 arrival times (QPS 4, 2, 1, 0.5)
bash experiments/crawler/run_scheduler_experiments.sh experiments/crawler/configs/H200_enhanced_schedulers_v1_full compare "0_25"
bash experiments/crawler/run_scheduler_experiments.sh experiments/crawler/configs/H200_enhanced_schedulers_v1_full compare "0_5"
bash experiments/crawler/run_scheduler_experiments.sh experiments/crawler/configs/H200_enhanced_schedulers_v1_full compare "1"
bash experiments/crawler/run_scheduler_experiments.sh experiments/crawler/configs/H200_enhanced_schedulers_v1_full compare "2"

