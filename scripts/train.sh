

accelerate launch --config_file scripts/ds_config.json \
    smes_ppo/train.py \
    --dataset_name jsonl_TrainValMerge_fullVidDescHis_10vidDescCurr_RL_max_emotion_max_strategy/train_val.jsonl \
    --dataset_train_split descriptiveness \
    --output_dir models/minimal/ppo \
    --num_ppo_epochs 1 \
    --num_mini_batches 1 \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --total_episodes 15000 \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --sft_model_path Qwen/Qwen2-0.5B-Instruct \
    --reward_model_path Qwen2-0.5B-Reward/checkpoint-410 \
    --local_rollout_forward_batch_size 1 \
    --missing_eos_penalty 1.0

