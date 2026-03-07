#!/bin/bash
#SBATCH -J H100_anns_full
#SBATCH --account=gts-krong6
#SBATCH -q inferno
#SBATCH -N 1
#SBATCH --mem=800GB
#SBATCH --ntasks-per-node=1
#SBATCH -t 36:00:00
#SBATCH --gres=gpu:H100:2
#SBATCH -o H100_anns_full_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rr@gatech.edu

CONDA_ENV=/storage/home/hcoda1/1/rbachkaniwala3/scratch/conda-envs/stream2llm
PROJECT_DIR=/storage/home/hcoda1/1/rbachkaniwala3/scratch/stream2llm

module load anaconda3
eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV

cd $PROJECT_DIR

# ANNS H100 experiment: all schedulers x 4 arrival times (QPS 2, 1, 0.5, 0.25)
bash experiments/anns/run_scheduler_experiments.sh experiments/anns/configs/H100_enhanced_schedulers_v1_full compare "0_5"
bash experiments/anns/run_scheduler_experiments.sh experiments/anns/configs/H100_enhanced_schedulers_v1_full compare "1"
bash experiments/anns/run_scheduler_experiments.sh experiments/anns/configs/H100_enhanced_schedulers_v1_full compare "2"
bash experiments/anns/run_scheduler_experiments.sh experiments/anns/configs/H100_enhanced_schedulers_v1_full compare "4"

