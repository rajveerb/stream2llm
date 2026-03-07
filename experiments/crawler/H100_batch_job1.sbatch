#!/bin/bash
#SBATCH -J H100_batch_job1           # Job name (anything you like)
#SBATCH --account=gts-krong6         # Your PACE account
#SBATCH -q inferno                   # QoS (inferno = paid, guaranteed)
#SBATCH -N 1                         # Number of nodes
#SBATCH --mem=500GB
#SBATCH --ntasks-per-node=1       # Tasks per node
#SBATCH -t 24:00:00                   # Walltime (hh:mm:ss)
#SBATCH --gres=gpu:H100:2            # Request 2x H100 GPUs on a single node
#SBATCH -o H100_batch_job_%j.out          # Output file (%j = JobID)
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rr@gatech.edu


CONDA_ENV=/storage/home/hcoda1/1/rbachkaniwala3/scratch/conda-envs/streamllm_rajveerb_H200
PROJECT_DIR=/storage/home/hcoda1/1/rbachkaniwala3/scratch/stream-llm_rajveerb_H200


# Load software modules
module load anaconda3
eval "$(conda shell.bash hook)"
# activate the conda environment
conda activate $CONDA_ENV
conda env list


cd $PROJECT_DIR

bash experiments/crawler/run_scheduler_experiments.sh experiments/crawler/configs/H100_enhanced_schedulers_v1 compare "0_0625"
bash experiments/crawler/run_scheduler_experiments.sh experiments/crawler/configs/H100_enhanced_schedulers_v1 compare "0_125"
bash experiments/crawler/run_scheduler_experiments.sh experiments/crawler/configs/H100_enhanced_schedulers_v1 compare "0_25"
bash experiments/crawler/run_scheduler_experiments.sh experiments/crawler/configs/H100_enhanced_schedulers_v1 compare "0_5"
bash experiments/crawler/run_scheduler_experiments.sh experiments/crawler/configs/H100_enhanced_schedulers_v1 compare "1"
bash experiments/crawler/run_scheduler_experiments.sh experiments/crawler/configs/H100_enhanced_schedulers_v1 compare "2"
bash experiments/crawler/run_scheduler_experiments.sh experiments/crawler/configs/H100_enhanced_schedulers_v1 compare "4"
bash experiments/crawler/generate_all_plots.sh data/run_log/crawler/H100_enhanced_schedulers_v1 experiments/crawler/analysis_results/

bash experiments/crawler/run_scheduler_experiments.sh experiments/crawler/configs/H100_enhanced_schedulers_v2 compare "0_0625"
bash experiments/crawler/run_scheduler_experiments.sh experiments/crawler/configs/H100_enhanced_schedulers_v2 compare "0_125"
bash experiments/crawler/run_scheduler_experiments.sh experiments/crawler/configs/H100_enhanced_schedulers_v2 compare "0_25"
bash experiments/crawler/run_scheduler_experiments.sh experiments/crawler/configs/H100_enhanced_schedulers_v2 compare "0_5"
bash experiments/crawler/run_scheduler_experiments.sh experiments/crawler/configs/H100_enhanced_schedulers_v2 compare "1"
bash experiments/crawler/run_scheduler_experiments.sh experiments/crawler/configs/H100_enhanced_schedulers_v2 compare "2"
bash experiments/crawler/run_scheduler_experiments.sh experiments/crawler/configs/H100_enhanced_schedulers_v2 compare "4"
bash experiments/crawler/generate_all_plots.sh data/run_log/crawler/H100_enhanced_schedulers_v2 experiments/crawler/analysis_results/