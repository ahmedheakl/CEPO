#!/bin/bash

set -x
export CUDA_VISIBLE_DEVICES=4,5,6,7
export NGPUS=4

MODEL_PATH=Qwen/Qwen3-VL-2B-Instruct
EXPERIMENT_NAME=qwen3_vl_2b_mmfine_rlsd_lora
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGS_NAME="${EXPERIMENT_NAME}_${TIMESTAMP}"

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=/data/fast0/users/ahmed_heakl/cepo/easyr1/data/mmfine/train.parquet \
    data.val_files=/data/fast0/users/ahmed_heakl/cepo/easyr1/data/mmfine/test.parquet \
    data.filter_overlong_prompts_workers=64 \
    data.rollout_batch_size=32 \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    algorithm.adv_estimator=rlsd \
    algorithm.disable_kl=True \
    algorithm.use_kl_loss=False \
    algorithm.rlsd_lambda_init=0.5 \
    algorithm.rlsd_lambda_decay_steps=25 \
    algorithm.rlsd_epsilon_w=0.2 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.model.lora.rank=16 \
    worker.actor.optim.lr=1e-6 \
    worker.actor.global_batch_size=16 \
    worker.actor.use_torch_compile=False \
    worker.actor.micro_batch_size_per_device_for_update=4 \
    worker.actor.micro_batch_size_per_device_for_experience=32 \
    worker.actor.fsdp.enable_full_shard=False \
    worker.actor.offload.offload_params=False \
    worker.actor.offload.offload_optimizer=False \
    worker.rollout.n=8 \
    worker.rollout.temperature=1.0 \
    worker.rollout.enforce_eager=False \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.max_model_len=4096 \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${NGPUS} \
    trainer.total_epochs=1 \
    trainer.max_steps=50 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="cepo" \
    trainer.val_freq=10 \
    trainer.val_before_train=True 2>&1 | tee "logs/${LOGS_NAME}"