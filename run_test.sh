#!/bin/bash

# -----------------------------------------------------------------
# >> CONFIGURATION (EDIT THESE VALUES) <<
# -----------------------------------------------------------------
#
# Set the (relative) path to your local folder containing .gguf models
LOCAL_MODEL_FOLDER="./llm-models"
#
# Set the simulated hardware limits
# This is where you configure your "mobile phone"
SIM_CPU_CORES="2.0"  # e.g., "1.5" (one and a half cores), "2.0", "4.0"
SIM_MEMORY="3g"      # e.g., "2g" (2GB), "4g" (4GB), "8g"

# -----------------------------------------------------------------
# >> BENCHMARK SCRIPT <<
# (You shouldn't need to edit below this line)
# -----------------------------------------------------------------

# Get the absolute path for the model folder
ABS_MODEL_PATH=$(realpath "$LOCAL_MODEL_FOLDER")
CONTAINER_MODEL_PATH="/models"

# Check if model folder exists
if [ ! -d "$ABS_MODEL_PATH" ]; then
    echo "Error: Model folder not found at: $ABS_MODEL_PATH"
    echo "Please update LOCAL_MODEL_FOLDER in this script."
    exit 1
fi

echo "--- Building Docker image: llm-bench ---"
docker build -t llm-bench .

echo ""
echo "--- Starting Benchmark Simulation ---"
echo "  CPUs:   $SIM_CPU_CORES"
echo "  Memory: $SIM_MEMORY"
echo "  Models: $ABS_MODEL_PATH"
echo "---------------------------------------"

# Run the docker container with the configured limits
docker run --rm \
  --cpus="$SIM_CPU_CORES" \
  --memory="$SIM_MEMORY" \
  -v "$ABS_MODEL_PATH":"$CONTAINER_MODEL_PATH" \
  llm-bench python3 /app/benchmark.py

echo "--- Benchmark Complete ---"