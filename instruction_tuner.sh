DATA_NAME_OR_PATH=
MODEL_NAME_OR_PATH=meta-llama/Llama-2-7b-hf
OUTPUT_DIR=
HUB_TOKEN=

export WANDB_API_KEY=
wandb login --cloud --host https://api.wandb.ai --relogin $WANDB_API_KEY

accelerate launch --config_file config.yaml instruction_tuner.py \
    --dataset_name_or_path $DATA_NAME_OR_PATH \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --use_flash_attention_2 \
    --hf_access_token $HUB_TOKEN \
    --torch_dtype bfloat16 \
    --max_seq_length 4096 \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --preprocessing_num_workers 12 \
    --seed 23 \
    --num_train_epochs 1 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --weight_decay 0.1 \
    --lr_scheduler_type cosine \
    --lr_warmup_fraction 0.01 \
    --with_tracking \
    --report_to wandb \
    --output_dir $OUTPUT_DIR \
    --push_to_hub \
    --hub_token $HUB_TOKEN \
    --private_repo