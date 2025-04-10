export PYTHONPATH=$PYTHONPATH:.
NUM_GPUS=2
NUM_NODES=1
NODE_RANK=0 
MASTER_ADDR="127.0.0.1"
MASTER_PORT=29500 

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=$NUM_NODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    reward_training/train.py \
    --deepspeed scripts/ds_config.json \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --dataset_name reward_train.json \
    --output_dir Qwen2-0.5B-Reward \
    --per_device_train_batch_size 1 \
    --num_train_epochs 10 \
    --gradient_checkpointing True \
    --learning_rate 2.5e-5 \
    --logging_steps 5 \
    --eval_strategy steps \
    --eval_steps 50 \
    --max_length 512 \
    --save_steps 10 \
    --ddp_find_unused_parameters False 
    

