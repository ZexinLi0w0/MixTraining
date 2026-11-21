#!/usr/bin/env bash
set -eu

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 MODEL TASK MIX_END_EPOCH [additional run_glue.py arguments]" >&2
    exit 2
fi

model=$1
task=$2
mix_epochs=$3
shift 3

case "$task" in
    cola) metric=matthews_correlation ;;
    stsb) metric=combined_score ;;
    *) metric=accuracy ;;
esac

exec python "$(dirname "$0")/run_glue.py" \
    --model_name_or_path "$model" --task_name "$task" \
    --do_train --do_eval --max_seq_length 128 \
    --per_device_train_batch_size 16 --learning_rate 2e-5 \
    --num_train_epochs 100 --mix_end_epoch "$mix_epochs" \
    --eval_strategy steps --save_strategy steps --load_best_model_at_end \
    --logging_steps 100 --save_steps 500 --eval_steps 100 \
    --metric_for_best_model "$metric" --save_total_limit 1 --report_to none \
    --output_dir "results/$task/${model##*/}/mix-$mix_epochs" "$@"
