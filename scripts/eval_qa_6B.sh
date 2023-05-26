export CUDA_VISIBLE_DEVICES=0

cd ..
for inputf in $(ls data/eval/qa/*.json); do
    python eval_qa.py \
        --data_path $inputf \
        --model_name_or_path EleutherAI/gpt-j-6B \
        --lora_name_or_path output/ewc-lora-6B/checkpoint \
        --output_log \
        --output_path output/ewc-lora-6B/qa-metric.txt
done