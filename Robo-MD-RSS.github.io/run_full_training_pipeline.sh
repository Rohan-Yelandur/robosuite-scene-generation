#!/bin/bash

# Full Training Pipeline Script
# This script runs:
# 1. train_discrete.py to collect perturbation data
# 2. train_embedding.py to learn embeddings from the collected data
# 3. train_continuous.py to train PPO with continuous latent actions

set -e  # Exit on error

# ============================================
# Configuration Variables
# ============================================
TASK_NAME="lift"  # Options: lift, can, square, stack, thread
AGENT_PATH="../policy/lift_ph_image_epoch_500_succ_100.pth"
RUN_NAME="lift_pipeline"

# Discrete RL parameters
DISCRETE_RL_TIMESTEPS=300
DISCRETE_RL_UPDATE_STEP=300
DISCRETE_SAVE_PATH="default_run"

# Embedding training parameters
EMBEDDING_EPOCHS=100
CHECKPOINT_GRADIENT=""  # Add "--checkpoint_gradient" to enable

# Continuous RL parameters
CONTINUOUS_RL_TIMESTEPS=3000

# ============================================
# Setup
# ============================================
echo "================================================"
echo "Starting Full Training Pipeline for task: $TASK_NAME"
echo "================================================"

# Activate conda environment
source ~/miniconda3/bin/activate robomd

# Create output directories
DISCRETE_DATA_DIR="${TASK_NAME}_rl_data/${DISCRETE_SAVE_PATH}"
mkdir -p "$DISCRETE_DATA_DIR"

# ============================================
# Step 1: Train Discrete RL to Collect Data
# ============================================
echo ""
echo "================================================"
echo "Step 1: Training Discrete RL Agent"
echo "================================================"
echo "Collecting perturbation data with different scene variations..."

# python3 train_discrete.py \
#     --task_name "$TASK_NAME" \
#     --agent_path "$AGENT_PATH" \
#     --rl_timesteps "$DISCRETE_RL_TIMESTEPS" \
#     --rl_update_step "$DISCRETE_RL_UPDATE_STEP" \
#     --save_path "$DISCRETE_SAVE_PATH" \
#     --collect_data

# if [ $? -eq 0 ]; then
#     echo "✓ Discrete RL training completed successfully"
#     echo "  Data saved to: $DISCRETE_DATA_DIR"
# else
#     echo "✗ Discrete RL training failed"
#     exit 1
# fi

# ============================================
# Step 2: Train Embedding Model
# ============================================
echo ""
echo "================================================"
echo "Step 2: Training Embedding Model"
echo "================================================"
echo "Learning embeddings from collected perturbation data..."

python3 train_embedding.py \
    --task "$TASK_NAME" \
    --path "$DISCRETE_DATA_DIR" \

if [ $? -eq 0 ]; then
    echo "✓ Embedding training completed successfully"
    echo "  Model saved to: embedding_model.pth"
    echo "  Embeddings saved to: known_embeddings.h5"
else
    echo "✗ Embedding training failed"
    exit 1
fi

# Check if embedding file was created
if [ ! -f "known_embeddings.h5" ]; then
    echo "✗ Error: known_embeddings.h5 not found!"
    exit 1
fi

# ============================================
# Step 3: Train Continuous RL with Latent Actions
# ============================================
echo ""
echo "================================================"
echo "Step 3: Training Continuous RL Agent"
echo "================================================"
echo "Training PPO with continuous latent actions..."

python3 train_continuous.py \
    --name "$RUN_NAME" \
    --task "$TASK_NAME" \
    --agent "$AGENT_PATH" \

if [ $? -eq 0 ]; then
    echo "✓ Continuous RL training completed successfully"
    echo "  Model saved to: ${RUN_NAME}/PPO_latent_${CONTINUOUS_RL_TIMESTEPS}.zip"
else
    echo "✗ Continuous RL training failed"
    exit 1
fi

# ============================================
# Summary
# ============================================
echo ""
echo "================================================"
echo "Training Pipeline Completed Successfully!"
echo "================================================"
echo ""
echo "Generated Files:"
echo "  1. Discrete data: $DISCRETE_DATA_DIR/demo/"
echo "  2. Embedding model: embedding_model.pth"
echo "  3. Known embeddings: known_embeddings.h5"
echo "  4. PPO model: ${RUN_NAME}/PPO_latent_${CONTINUOUS_RL_TIMESTEPS}.zip"
echo ""
echo "Logs:"
echo "  - Discrete RL: ${TASK_NAME}_logs/${DISCRETE_SAVE_PATH}/"
echo "  - Continuous RL: ${RUN_NAME}/latent_ppo_logs/"
echo ""


