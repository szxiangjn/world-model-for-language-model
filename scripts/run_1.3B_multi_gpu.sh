cd ..
accelerate launch --config_file scripts/accelerate_config.json run.py \
	--train_file "1 data/train/train_plan_gen.json 0.7 data/train/train_act_recog.json 1 data/train/train_count.json 1 data/train/train_obj_move.json" \
	--fisher_matrix_path fisher-matrix/fisher-matrix-1.3B \
	--model_name_or_path EleutherAI/gpt-neo-1.3B \
	--per_device_train_batch_size 10 \
	--lr 8e-4 \
	--output_dir output/ewc-lora-1.3B/checkpoint \
	--num_epochs 3 \
	--ewc_lambda 2
