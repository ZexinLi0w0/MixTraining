TASK_NAME=qnli
MODEL=gpt2
# MIX_END_EPOCH=0

for MIX_END_EPOCH in 0 25; do
  CUDA_VISIBLE_DEVICES=4 python run_glue.py \
    --model_name_or_path $MODEL \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --max_seq_length 128 \
    --per_device_train_batch_size 16 \
    --evaluation_strategy steps \
    --save_strategy steps \
    --load_best_model_at_end \
    --metric_for_best_model accuracy \
    --logging_strategy steps \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --num_train_epochs 100 \
    --mix_end_epoch $MIX_END_EPOCH \
    --report_to wandb \
    --logging_steps 100 \
    --pad_to_max_length \
    --run_name run_glue_${TASK_NAME}_${MODEL}_mix-${MIX_END_EPOCH} \
    --output_dir results/$TASK_NAME/$MODEL/mix-$MIX_END_EPOCH \
    --overwrite_output_dir
done