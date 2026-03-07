#!/bin/bash

# Replay rate variation script
# This script will vary the replay rate and run the crawler for each variation
# Usage: bash experiments/crawler/vary_replay_rate.sh <config_type> <base_config_dir>
# config_type must be one of: recomp, swap, or recomp_and_swap
# base_config_dir is required and should point to the base directory (e.g., configs/aws_A100_v2)
#
# Example:
#   bash experiments/crawler/vary_replay_rate.sh recomp configs/aws_A100_v2

DIR="experiments/crawler"
CONFIG_TYPE=$1
BASE_CONFIG_DIR=$2
# remove trailing slash from BASE_CONFIG_DIR if it exists
BASE_CONFIG_DIR=$(echo "$BASE_CONFIG_DIR" | sed 's:/*$::')
CONFIG_DIR="${BASE_CONFIG_DIR}/${CONFIG_TYPE}"
CONFIG_FILE_PREFIX="config_replay_web_crawl_replay_"

# Array of replay rates to test
avg_arrival_rates=("0_0625" "0_125" "0_25" "0_5" "1" "2" "4" "8" "16" "32")

# Validate base config directory is provided
if [ -z "$BASE_CONFIG_DIR" ]; then
    echo "Error: Base config directory is required"
    echo "Usage: bash experiments/crawler/vary_replay_rate.sh <config_type> <base_config_dir>"
    echo ""
    echo "Example:"
    echo "  bash experiments/crawler/vary_replay_rate.sh recomp configs/aws_A100_v2"
    exit 1
fi

# Validate config type
if [[ ! "$CONFIG_TYPE" =~ ^(recomp|swap|recomp_and_swap)$ ]]; then
    echo "Error: Invalid config type '$CONFIG_TYPE'"
    echo "Usage: bash experiments/crawler/vary_replay_rate.sh <config_type> <base_config_dir>"
    echo "config_type must be one of: recomp, swap, or recomp_and_swap"
    echo ""
    echo "Example:"
    echo "  bash experiments/crawler/vary_replay_rate.sh recomp configs/aws_A100_v2"
    exit 1
fi

# Validate config directory exists
if [ ! -d "$DIR/$CONFIG_DIR" ]; then
    echo "Error: Config directory '$DIR/$CONFIG_DIR' does not exist"
    exit 1
fi

# print conda env and machine name
echo "Running on machine: $(hostname)"
echo "Running in conda environment: $CONDA_DEFAULT_ENV"
echo "Using config type: $CONFIG_TYPE"

# Loop through each rate
for avg_arrival_rate in "${avg_arrival_rates[@]}"; do
    config_file="$DIR/$CONFIG_DIR/${CONFIG_FILE_PREFIX}${avg_arrival_rate}.yaml"
    
    # Validate config file exists
    if [ ! -f "$config_file" ]; then
        echo "Error: Config file '$config_file' does not exist"
        exit 1
    fi
    
    echo "======================================================"
    echo "Starting replay with rate: $avg_arrival_rate secs per query"
    echo "Running: python $DIR/replay_query_crawls.py --config $config_file"
    echo "======================================================"
    
    # Run the crawler with the current rate
    python $DIR/replay_query_crawls.py --config "$config_file"
    
    echo "Completed replay with rate: $avg_arrival_rate secs per query"
    echo ""
done

echo "All replay rates completed!"
