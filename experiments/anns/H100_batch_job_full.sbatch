#!/bin/bash
#SBATCH -J H100_anns_batch_full           # Job name (anything you like)
#SBATCH --account=gts-krong6-paid         # Your PACE account
#SBATCH -q inferno                   # QoS (inferno = paid, guaranteed)
#SBATCH -N 1                         # Number of nodes
#SBATCH --mem=800GB
#SBATCH --ntasks-per-node=1       # Tasks per node
#SBATCH -t 36:00:00                   # Walltime (hh:mm:ss)
#SBATCH --gres=gpu:H100:2            # Request 2x H100 GPUs on a single node
#SBATCH -o H100_anns_batch_full_%j.out          # Output file (%j = JobID)
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rr@gatech.edu


CONDA_ENV=/storage/home/hcoda1/1/rbachkaniwala3/p-krong6-0/conda-envs/stream_llm_main
PROJECT_DIR=/storage/home/hcoda1/1/rbachkaniwala3/p-krong6-0/stream-llm_rajveerb


# Load software modules
module load anaconda3
eval "$(conda shell.bash hook)"
# activate the conda environment
conda activate $CONDA_ENV
conda env list


cd $PROJECT_DIR

# Run ANNS experiments for H100 enhanced schedulers across different arrival times
bash experiments/anns/run_scheduler_experiments.sh experiments/anns/configs/H100_enhanced_schedulers_v1_full compare "0_0625"
bash experiments/anns/run_scheduler_experiments.sh experiments/anns/configs/H100_enhanced_schedulers_v1_full compare "0_125"
bash experiments/anns/run_scheduler_experiments.sh experiments/anns/configs/H100_enhanced_schedulers_v1_full compare "0_25"
bash experiments/anns/run_scheduler_experiments.sh experiments/anns/configs/H100_enhanced_schedulers_v1_full compare "0_5"
bash experiments/anns/run_scheduler_experiments.sh experiments/anns/configs/H100_enhanced_schedulers_v1_full compare "1"
bash experiments/anns/run_scheduler_experiments.sh experiments/anns/configs/H100_enhanced_schedulers_v1_full compare "2"
bash experiments/anns/run_scheduler_experiments.sh experiments/anns/configs/H100_enhanced_schedulers_v1_full compare "4"

# Generate analysis plots if analysis tools exist
bash experiments/anns/generate_all_plots.sh data/run_log/anns/H100_enhanced_schedulers_v1_full experiments/anns/analysis_results/