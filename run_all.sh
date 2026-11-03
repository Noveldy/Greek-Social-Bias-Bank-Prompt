#!/bin/bash
# Run all the revision runs on the Arc GPU, one model at a time.
# Safe to restart at any time (e.g. after a power cut): finished runs are skipped and an interrupted run
# continues from the first prompt without an answer.
# Usage: bash run_all.sh      (progress: results/logs/run_all.log, details: results/logs/inference_<model>_<run>.log)
cd "$(dirname "$0")"
if pgrep -f "run_inference.py" > /dev/null; then
    echo "A run is already in progress, not starting a second one"
    exit 1
fi
[ -z "$ONEAPI_ROOT" ] && source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1

MODELS="Qwen3-4B-Instruct-2507-Q5_K_M llama-krikri-8b-instruct-q5_k_m gemma-4-E2B-it-Q5_K_M"
RUNS="v2_chat name_swap_chat paraphrase_chat option_order_chat sampling_seed1_chat sampling_seed2_chat sampling_seed3_chat"
QUEUE_LOG=results/logs/run_all.log
mkdir -p results/logs

for run in $RUNS; do
    for model in $MODELS; do
        log=results/logs/inference_${model}_${run}.log
        echo "$(date '+%F %T') start $model $run" | tee -a $QUEUE_LOG
        if ! .venv/bin/python run_inference.py $model $run >> $log 2>&1; then
            echo "$(date '+%F %T') FAILED $model $run, see $log" | tee -a $QUEUE_LOG
            exit 1
        fi
        answers=$(($(grep -c . output/llm_responses_${model}_${run}.csv) - 1))
        echo "$(date '+%F %T') done  $model $run: $answers answers, $(grep -o 'offloaded [0-9/]* layers to GPU' $log | tail -1)" | tee -a $QUEUE_LOG
    done
done
echo "$(date '+%F %T') ALL DONE" | tee -a $QUEUE_LOG
