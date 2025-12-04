#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

run_training() {
    local task_name="$1"
    local total_steps="$2"

    echo "============================================================"
    echo "[RUN] Training ${task_name} for ${total_steps} timesteps"
    echo "============================================================"

    python train_ppo.py \
        --env "${task_name}" \
        --timesteps "${total_steps}" \
        --chunk_size 250000 \
        --num_cpu 8

    echo "[DONE] ${task_name} training complete"
    echo
}

run_training "Lift" 2000000
run_training "Can" 4000000
run_training "Square" 8000000

echo "All requested trainings completed."
