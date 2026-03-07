#!/bin/bash
#SBATCH -J H200_crawler_delay_10_recomp_only
#SBATCH --account=gts-krong6
#SBATCH -q inferno
#SBATCH -N 1
#SBATCH --mem=800GB
#SBATCH --ntasks-per-node=1
#SBATCH -t 8:00:00
#SBATCH --gres=gpu:H200:2
#SBATCH -o H200_crawler_delay_10_recomp_only_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rr@gatech.edu

CONDA_ENV=/storage/home/hcoda1/1/rbachkaniwala3/scratch/conda-envs/stream2llm
PROJECT_DIR=/storage/home/hcoda1/1/rbachkaniwala3/scratch/stream2llm

module load anaconda3
eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV

cd $PROJECT_DIR

# Crawler recomp-only ablation (delay_10): all schedulers at QPS 4
bash experiments/crawler/run_scheduler_experiments.sh experiments/crawler/configs/H200_enhanced_schedulers_v1_full_delay_10_recomp_only compare "0_25"
