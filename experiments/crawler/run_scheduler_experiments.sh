#!/bin/bash

# Enhanced Scheduler Experiment Runner
# This script runs experiments for all scheduler types across different arrival rates

set -e
# Check for help command first
if [ "$1" = "help" ] || [ "$#" -eq 0 ]; then
    BASE_CONFIG_DIR=""
elif [ -n "$1" ]; then
    BASE_CONFIG_DIR="$1"
    shift  # Remove base config dir from arguments
else
    echo "Usage: $0 <base_config_directory> <command> [arguments] OR $0 help"
    exit 1
fi


# Configuration
#BASE_CONFIG_DIR set above
SCHEDULERS=("default_vllm" "fcfs_lru" "lcas_lifo" "mcps_lce" "oeda_pbas" "stream_based_v1" "lcas_cplusp")
ARRIVAL_TIMES=("0_0625" "0_125" "0_25" "0_5" "1" "2" "4" "8" "16" "32")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}ENHANCED SCHEDULER EXPERIMENT RUNNER${NC}"
echo -e "${BLUE}========================================${NC}"

# Function to run a single experiment
run_experiment() {
    local scheduler=$1
    local arrival_time=$2
    # Convert dots to underscores for filename (e.g., 0.125 -> 0_125)
    local filename_time=$(echo "$arrival_time" | sed 's/\./_/g')
    local config_file="${BASE_CONFIG_DIR}/${scheduler}/config_replay_web_crawl_replay_${filename_time}.yaml"
    
    if [ ! -f "$config_file" ]; then
        echo -e "${RED}Config file not found: $config_file. Skipping.${NC}"
        return 0
    fi
    
    echo -e "\n${YELLOW}Running: ${scheduler} with arrival time ${arrival_time}s${NC}"
    echo -e "${BLUE}Config: $config_file${NC}"
    
    # Note: SCHEDULER_TYPE is set from config file, not here
    
    # Run the experiment
    python experiments/crawler/replay_query_crawls.py --config "$config_file"
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}✓ Completed: ${scheduler} - ${arrival_time}s${NC}"
    else
        echo -e "${RED}✗ Failed: ${scheduler} - ${arrival_time}s${NC}"
        return $exit_code
    fi
}

# Function to run all experiments for a single scheduler
run_scheduler_experiments() {
    local scheduler=$1
    echo -e "\n${GREEN}======= RUNNING EXPERIMENTS FOR: ${scheduler} =======${NC}"
    
    for arrival_time in "${ARRIVAL_TIMES[@]}"; do
        run_experiment "$scheduler" "$arrival_time"
    done
}

# Function to run all experiments for a single arrival time
run_arrival_time_experiments() {
    local arrival_time=$1
    echo -e "\n${GREEN}======= RUNNING EXPERIMENTS FOR ARRIVAL TIME: ${arrival_time}s =======${NC}"
    for scheduler in "${SCHEDULERS[@]}"; do
        run_experiment "$scheduler" "$arrival_time"
    done
}

# Function to run a specific experiment
run_single_experiment() {
    local scheduler=$1
    local arrival_time=$2
    
    echo -e "\n${GREEN}======= RUNNING SINGLE EXPERIMENT =======${NC}"
    echo -e "${GREEN}Scheduler: $scheduler${NC}"
    echo -e "${GREEN}Arrival Time: ${arrival_time}s${NC}"
    
    run_experiment "$scheduler" "$arrival_time"
}

# Function to show available configurations
show_configs() {
    echo -e "\n${BLUE}Available Scheduler Configurations:${NC}"
    for scheduler in "${SCHEDULERS[@]}"; do
        echo -e "${YELLOW}$scheduler:${NC}"
        config_count=$(ls -1 "${BASE_CONFIG_DIR}/${scheduler}/"*.yaml 2>/dev/null | wc -l)
        echo -e "  Configurations: $config_count"
        echo -e "  Directory: ${BASE_CONFIG_DIR}/${scheduler}/"
        echo ""
    done
    
    echo -e "${BLUE}Available Arrival Times (seconds):${NC}"
    for time in "${ARRIVAL_TIMES[@]}"; do
        qps=$(echo "scale=4; 1 / ${time//_/.}" | bc -l 2>/dev/null || echo "N/A")
        echo -e "  ${time} seconds (${qps} QPS)"
    done
}

# Function to compare schedulers at specific load
run_comparison() {
    local arrival_time=$1
    
    echo -e "\n${GREEN}======= COMPARING ALL SCHEDULERS AT ${arrival_time}s =======${NC}"
    
    for scheduler in "${SCHEDULERS[@]}"; do
        echo -e "\n${YELLOW}Running $scheduler...${NC}"
        run_experiment "$scheduler" "$arrival_time"
        sleep 1
    done
    
    echo -e "\n${GREEN}✓ Comparison complete for arrival time: ${arrival_time}s${NC}"
}

# Main execution logic
case "${1:-help}" in
    "all")
        echo -e "${GREEN}Running ALL arrival time experiments...${NC}"
        for arrival_time in "${ARRIVAL_TIMES[@]}"; do
            run_arrival_time_experiments "$arrival_time"
            echo -e "${GREEN}Completed all experiments for arrival time: ${arrival_time}s${NC}\n"
        done
        ;;
        
    "scheduler")
        if [ -z "$2" ]; then
            echo -e "${RED}Usage: $0 scheduler <scheduler_name>${NC}"
            echo -e "${BLUE}Available schedulers: ${SCHEDULERS[*]}${NC}"
            exit 1
        fi
        run_scheduler_experiments "$2"
        ;;
        
    "single")
        if [ -z "$2" ] || [ -z "$3" ]; then
            echo -e "${RED}Usage: $0 single <scheduler_name> <arrival_time>${NC}"
            echo -e "${BLUE}Available schedulers: ${SCHEDULERS[*]}${NC}"
            echo -e "${BLUE}Available arrival times: ${ARRIVAL_TIMES[*]}${NC}"
            exit 1
        fi
        run_single_experiment "$2" "$3"
        ;;
        
    "compare")
        if [ -z "$2" ]; then
            echo -e "${RED}Usage: $0 compare <arrival_time>${NC}"
            echo -e "${BLUE}Available arrival times: ${ARRIVAL_TIMES[*]}${NC}"
            exit 1
        fi
        run_comparison "$2"
        ;;
        
    "configs")
        show_configs
        ;;
        
    "help"|*)
        cat << HELP

Enhanced Scheduler Experiment Runner

USAGE:
    $0 help                                 # Show this help
    $0 <base_config_directory> <command> [arguments]

COMMANDS:
    all                           - Run ALL experiments across ALL arrival times
    scheduler <name>              - Run all arrival times for specific scheduler
    single <scheduler> <time>     - Run single experiment
    compare <arrival_time>        - Compare all schedulers at specific load
    configs                       - Show available configurations
    help                          - Show this help

EXAMPLES:
    $0 <config_dir> all                        # Run everything (will take a long time!)
    $0 <config_dir> scheduler fcfs_lru         # Run all experiments for FCFS-LRU
    $0 <config_dir> single oeda_pbas 0_125     # Run OEDA-PBAS at 8 QPS (0.125s arrival)
    $0 <config_dir> compare 0_25               # Compare all schedulers at 4 QPS
    $0 <config_dir> configs                    # List all available configurations

AVAILABLE SCHEDULERS:
    default_vllm     - Standard vLLM scheduler
    fcfs_lru         - First-Come-First-Served with LRU eviction
    lcas_lifo        - Last Chunk Arrival with LIFO eviction  
    mcps_lce         - Most Chunks Processed with Least Chunks eviction
    oeda_pbas        - Deadline-aware priority-based scheduling
    stream_based_v1  - Stream-aware scheduling

ARRIVAL TIMES (seconds -> QPS):
    0_0625 or 0.0625 -> 16 QPS    0_125 or 0.125 -> 8 QPS     0_25 or 0.25 -> 4 QPS      
    0_5 or 0.5 -> 2 QPS           1 -> 1 QPS                  2 -> 0.5 QPS       
    4 -> 0.25 QPS                 8 -> 0.125 QPS              16 -> 0.0625 QPS    
    32 -> 0.03125 QPS

HELP
        ;;
esac

echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}Experiment runner completed.${NC}"
echo -e "${BLUE}========================================${NC}"
