export CUDA_VISIBLE_DEVICES=0 

cd ..
python run.py \
	--train_file "1 data/train/train_plan_gen.json 0.7 data/train/train_act_recog.json 1 data/train/train_count.json 1 data/train/train_obj_move.json" \
	--fisher_matrix_path fisher-matrix/fisher-matrix-6B \
	--model_name_or_path EleutherAI/gpt-j-6B \
	--per_device_train_batch_size 5 \
	--gradient_accumulation_steps 4 \
	--lr 8e-4 \
	--output_dir output/ewc-lora-6B/checkpoint \
	--num_epochs 5 \
	--ewc_lambda 0.5
