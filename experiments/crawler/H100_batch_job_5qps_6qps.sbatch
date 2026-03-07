#!/bin/bash
#SBATCH -J H100_5qps_6qps           # Job name (anything you like)
#SBATCH --account=gts-krong6         # Your PACE account
#SBATCH -q inferno                   # QoS (inferno = paid, guaranteed)
#SBATCH -N 1                         # Number of nodes
#SBATCH --mem=500GB
#SBATCH --ntasks-per-node=1       # Tasks per node
#SBATCH -t 14:00:00                   # Walltime (hh:mm:ss)
#SBATCH --gres=gpu:H100:2            # Request 2x H100 GPUs on a single node
#SBATCH -o H100_5qps_6qps_%j.out          # Output file (%j = JobID)
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rr@gatech.edu


CONDA_ENV=/storage/home/hcoda1/1/rbachkaniwala3/scratch/conda-envs/stream2llm_rajveerb_H200
PROJECT_DIR=/storage/home/hcoda1/1/rbachkaniwala3/scratch/stream-llm_rajveerb_H200


# Load software modules
module load anaconda3
eval "$(conda shell.bash hook)"
# activate the conda environment
conda activate $CONDA_ENV
conda env list


cd $PROJECT_DIR

# Run experiments at 5 QPS (0.2s arrival time)
bash experiments/crawler/run_scheduler_experiments.sh experiments/crawler/configs/H100_enhanced_schedulers_v1_full compare "0_2"

# Run experiments at 6 QPS (0.166667s arrival time)
bash experiments/crawler/run_scheduler_experiments.sh experiments/crawler/configs/H100_enhanced_schedulers_v1_full compare "0_166667"

# Generate plots
bash experiments/crawler/generate_all_plots.sh data/run_log/crawler/H100_enhanced_schedulers_v1_full experiments/crawler/analysis_results/
