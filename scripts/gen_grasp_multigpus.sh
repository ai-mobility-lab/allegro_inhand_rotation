#!/bin/bash
# Multi-GPU grasp generation script
# Distributes scale generations across multiple GPUs sequentially per GPU
# Usage: ./gen_grasp_multigpus.sh GPU1 GPU2 [GPU3 ...]
# Example: ./gen_grasp_multigpus.sh 0 1

if [ $# -lt 1 ]; then
    echo "Error: At least one GPU number is required"
    echo "Usage: ./gen_grasp_multigpus.sh GPU1 GPU2 [GPU3 ...]"
    echo "Example: ./gen_grasp_multigpus.sh 0 1"
    exit 1
fi

GPUS=($@)
NUM_GPUS=${#GPUS[@]}

# Comprehensive scale list covering training range and boundaries
# Training uses: [0.7, 0.72, 0.74, 0.76, 0.78, 0.8, 0.82, 0.84, 0.86]
# We add boundary values 0.66 and 0.9 for robustness
SCALES=(0.66 0.7 0.72 0.74 0.76 0.78 0.8 0.82 0.84 0.86 0.9)
NUM_SCALES=${#SCALES[@]}

echo "========================================="
echo "Multi-GPU Grasp Generation (Sequential per GPU)"
echo "GPUs: ${GPUS[@]} (${NUM_GPUS} GPUs)"
echo "Scales: ${SCALES[@]} (${NUM_SCALES} scales)"
echo ""
echo "Distribution (Round-Robin, Sequential per GPU):"
for i in "${!SCALES[@]}"; do
    SCALE=${SCALES[$i]}
    GPU=${GPUS[$((i % NUM_GPUS))]}
    echo "  GPU ${GPU}: SCALE ${SCALE}"
done
echo "========================================="
echo ""

# Function to run scales sequentially on a specific GPU
run_gpu_worker() {
    local GPU=$1
    shift
    local SCALES_FOR_GPU=("$@")

    for SCALE in "${SCALES_FOR_GPU[@]}"; do
        echo "[GPU ${GPU}] Starting SCALE ${SCALE}..."

        CUDA_VISIBLE_DEVICES=${GPU} \
        python gen_grasp.py task=LeftAllegroHandGrasp headless=True pipeline=cpu \
        task.env.numEnvs=20000 test=True \
        task.env.controller.controlFrequencyInv=8 task.env.episodeLength=50 \
        task.env.controller.torque_control=False task.env.genGrasps=True task.env.baseObjScale="${SCALE}" \
        task.env.object.type=simple_tennis_ball task.env.object.sampleProb=[1.0] \
        task.env.randomization.randomizeMass=True task.env.randomization.randomizeMassLower=0.05 task.env.randomization.randomizeMassUpper=0.051 \
        task.env.randomization.randomizeCOM=False \
        task.env.randomization.randomizeFriction=False \
        task.env.randomization.randomizePDGains=False \
        task.env.randomization.randomizeScale=False \
        train.ppo.priv_info=True \
        > "gen_grasp_scale_${SCALE}_gpu_${GPU}.log" 2>&1

        EXIT_CODE=$?
        if [ $EXIT_CODE -eq 0 ]; then
            echo "[GPU ${GPU}] ✓ SCALE ${SCALE} completed"
        else
            echo "[GPU ${GPU}] ✗ SCALE ${SCALE} failed (exit code ${EXIT_CODE})"
            return $EXIT_CODE
        fi
    done

    return 0
}

# Distribute scales to GPUs
declare -A GPU_SCALES
for GPU in "${GPUS[@]}"; do
    GPU_SCALES[$GPU]=""
done

for i in "${!SCALES[@]}"; do
    SCALE=${SCALES[$i]}
    GPU=${GPUS[$((i % NUM_GPUS))]}
    GPU_SCALES[$GPU]+="${SCALE} "
done

# Launch one worker process per GPU
PIDS=()
for GPU in "${GPUS[@]}"; do
    echo "Launching worker for GPU ${GPU}..."
    run_gpu_worker $GPU ${GPU_SCALES[$GPU]} &
    PIDS+=($!)
    sleep 2  # Stagger startup to avoid initialization conflicts
done

echo ""
echo "All GPU workers launched. PIDs: ${PIDS[@]}"
echo ""

# Wait for all workers to complete
FAILED=0
for i in "${!PIDS[@]}"; do
    PID=${PIDS[$i]}
    GPU=${GPUS[$i]}

    wait $PID
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ GPU ${GPU} worker completed successfully"
    else
        echo "✗ GPU ${GPU} worker failed"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "========================================="
echo "Summary:"
echo "  Total scales processed: ${NUM_SCALES}"
echo "  GPUs used: ${NUM_GPUS}"
echo "  Scales per GPU: ~$((NUM_SCALES / NUM_GPUS)) - $(((NUM_SCALES + NUM_GPUS - 1) / NUM_GPUS))"
echo ""
if [ $FAILED -eq 0 ]; then
    echo "✓ All ${NUM_SCALES} scales completed successfully!"
    echo ""
    echo "Generated files in cache/:"
    ls -lh cache/allegro_right_grasp_50k_s*.npy | tail -${NUM_SCALES}
else
    echo "✗ ${FAILED} GPU worker(s) failed"
    echo "  Check gen_grasp_scale_*.log for details"
fi
echo "========================================="

exit $FAILED
