export CUDA_VISIBLE_DEVICES=0

cd ..
for inputf in $(ls data/eval/qa/*.json); do
    python eval_qa.py \
        --data_path $inputf \
        --model_name_or_path EleutherAI/gpt-neo-1.3B \
        --lora_name_or_path output/ewc-lora-1.3B/checkpoint \
        --output_log \
        --output_path output/ewc-lora-1.3B/qa-metric.txt
done