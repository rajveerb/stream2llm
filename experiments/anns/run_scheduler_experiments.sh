#!/bin/bash

# ANNS Enhanced Scheduler Experiment Runner
# This script runs ANNS experiments for all scheduler types across different arrival times

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
SCHEDULERS=("default_vllm" "fcfs_lru" "lcas_cplusp" "mcps_lce")
ARRIVAL_TIMES=("0_0625" "0_125" "0_25" "0_5" "1" "2" "4")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}ANNS ENHANCED SCHEDULER EXPERIMENT RUNNER${NC}"
echo -e "${BLUE}========================================${NC}"

# Function to run a single experiment
run_experiment() {
    local scheduler=$1
    local arrival_time=$2
    local config_file="${BASE_CONFIG_DIR}/${scheduler}/config_replay_anns_replay_${arrival_time}.yaml"

    if [ ! -f "$config_file" ]; then
        echo -e "${RED}Config file not found: $config_file. Skipping.${NC}"
        return 0
    fi

    # Convert arrival time to actual value for display
    local arrival_value=$(echo "$arrival_time" | sed 's/_/./g')
    local qps_value=$(echo "scale=4; 1 / $arrival_value" | bc -l 2>/dev/null || echo "N/A")

    echo -e "\n${YELLOW}Running: ${scheduler} at ${arrival_value}s arrival time (${qps_value} QPS)${NC}"
    echo -e "${BLUE}Config: $config_file${NC}"

    # Run the experiment
    python experiments/anns/replay_anns_queries.py --config "$config_file"

    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}✓ Completed: ${scheduler} - ${arrival_value}s${NC}"
    else
        echo -e "${RED}✗ Failed: ${scheduler} - ${arrival_value}s${NC}"
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
    local arrival_value=$(echo "$arrival_time" | sed 's/_/./g')
    local qps_value=$(echo "scale=4; 1 / $arrival_value" | bc -l 2>/dev/null || echo "N/A")
    echo -e "\n${GREEN}======= RUNNING EXPERIMENTS FOR ARRIVAL TIME: ${arrival_value}s (${qps_value} QPS) =======${NC}"

    for scheduler in "${SCHEDULERS[@]}"; do
        run_experiment "$scheduler" "$arrival_time"
    done
}

# Function to run a specific experiment
run_single_experiment() {
    local scheduler=$1
    local arrival_time=$2
    local arrival_value=$(echo "$arrival_time" | sed 's/_/./g')

    echo -e "\n${GREEN}======= RUNNING SINGLE EXPERIMENT =======${NC}"
    echo -e "${GREEN}Scheduler: $scheduler${NC}"
    echo -e "${GREEN}Arrival Time: ${arrival_value}s${NC}"

    run_experiment "$scheduler" "$arrival_time"
}

# Function to show available configurations
show_configs() {
    echo -e "\n${BLUE}Available ANNS Scheduler Configurations:${NC}"
    for scheduler in "${SCHEDULERS[@]}"; do
        echo -e "${YELLOW}$scheduler:${NC}"
        config_count=$(ls -1 "${BASE_CONFIG_DIR}/${scheduler}/"*.yaml 2>/dev/null | wc -l)
        echo -e "  Configurations: $config_count"
        echo -e "  Directory: ${BASE_CONFIG_DIR}/${scheduler}/"
        echo ""
    done

    echo -e "${BLUE}Available Arrival Times:${NC}"
    for arrival_time in "${ARRIVAL_TIMES[@]}"; do
        arrival_value=$(echo "$arrival_time" | sed 's/_/./g')
        qps_value=$(echo "scale=4; 1 / $arrival_value" | bc -l 2>/dev/null || echo "N/A")
        echo -e "  ${arrival_value}s (${qps_value} QPS)"
    done
}

# Function to compare schedulers at specific arrival time
run_comparison() {
    local arrival_time=$1
    local arrival_value=$(echo "$arrival_time" | sed 's/_/./g')
    local qps_value=$(echo "scale=4; 1 / $arrival_value" | bc -l 2>/dev/null || echo "N/A")

    echo -e "\n${GREEN}======= COMPARING ALL SCHEDULERS AT ${arrival_value}s (${qps_value} QPS) =======${NC}"

    for scheduler in "${SCHEDULERS[@]}"; do
        echo -e "\n${YELLOW}Running $scheduler...${NC}"
        run_experiment "$scheduler" "$arrival_time"
        sleep 1
    done

    echo -e "\n${GREEN}✓ Comparison complete for arrival time: ${arrival_value}s${NC}"
}

# Main execution logic
case "${1:-help}" in
    "all")
        echo -e "${GREEN}Running ALL arrival time experiments...${NC}"
        for arrival_time in "${ARRIVAL_TIMES[@]}"; do
            run_arrival_time_experiments "$arrival_time"
            echo -e "${GREEN}Completed all experiments for arrival time: $(echo "$arrival_time" | sed 's/_/./g')s${NC}\n"
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

ANNS Enhanced Scheduler Experiment Runner

USAGE:
    $0 help                                 # Show this help
    $0 <base_config_directory> <command> [arguments]

COMMANDS:
    all                           - Run ALL experiments across ALL arrival times
    scheduler <name>              - Run all arrival times for specific scheduler
    single <scheduler> <qps>      - Run single experiment
    compare <qps_load>           - Compare all schedulers at specific QPS
    configs                       - Show available configurations
    help                          - Show this help

EXAMPLES:
    $0 experiments/anns/configs all                        # Run everything (will take a long time!)
    $0 experiments/anns/configs scheduler fcfs_lru         # Run all experiments for FCFS-LRU
    $0 experiments/anns/configs single mcps_lce 4          # Run MCPS-LCE at 4 QPS
    $0 experiments/anns/configs compare 1                  # Compare all schedulers at 1 QPS
    $0 experiments/anns/configs configs                    # List all available configurations

AVAILABLE SCHEDULERS:
    default_vllm     - Standard vLLM scheduler
    fcfs_lru         - First-Come-First-Served with LRU eviction
    lcas_cplusp      - Last Chunk Arrival with C++ eviction
    mcps_lce         - Most Chunks Processed with Least Chunks eviction

QPS LOADS:
    0.25 QPS     0.5 QPS     1 QPS      2 QPS
    4 QPS        8 QPS       16 QPS

NOTES:
    - Each experiment processes 500 queries
    - Poisson arrival times are pre-generated for reproducibility
    - Results are saved in data/run_log/anns/enhanced_schedulers/<scheduler>/
    - Performance models (swap/recomputation) are enabled for non-default schedulers

HELP
        ;;
esac

echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}ANNS experiment runner completed.${NC}"
echo -e "${BLUE}========================================${NC}"