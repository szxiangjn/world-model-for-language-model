export CUDA_VISIBLE_DEVICES=0

cd ..
for inputf in $(ls data/eval/gen/*.json); do
    task_name=$(basename $inputf .json)
    METRIC="rouge"
    if [ $task_name == "obj_move_path" ]; then
        METRIC="lcs"
    fi
    python eval_gen.py \
        --val_file $inputf \
        --model_name_or_path EleutherAI/gpt-j-6B \
        --metric_name_or_path $METRIC \
        --lora_name_or_path output/ewc-lora-6B/checkpoint \
        --max_tgt_len 256 \
        --per_device_eval_batch_size 2 \
        --output_log 1 \
        --output_path output/ewc-lora-6B/gen-metric.txt
done