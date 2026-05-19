#!/bin/bash

set -x
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NGPUS=8

MODEL_PATH=Qwen/Qwen3-VL-2B-Instruct
EXPERIMENT_NAME=qwen3_vl_2b_geo_grpo_lora16_200steps
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGS_NAME="${EXPERIMENT_NAME}_${TIMESTAMP}"
python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=hiyouga/geometry3k@train \
    data.val_files=hiyouga/geometry3k@test \
    data.filter_overlong_prompts_workers=64 \
    data.rollout_batch_size=32 \
    data.max_prompt_length=768 \
    data.max_response_length=2048 \
    data.format_prompt=../../examples/format_prompt/math_short.jinja \
    algorithm.adv_estimator=grpo \
    algorithm.disable_kl=True \
    algorithm.use_kl_loss=False \
    algorithm.online_filtering=False \
    worker.actor.clip_ratio_low=0.2 \
    worker.actor.clip_ratio_high=0.28 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.model.lora.rank=16 \
    worker.actor.optim.lr=1e-6 \
    worker.actor.global_batch_size=16 \
    worker.actor.use_torch_compile=False \
    worker.actor.micro_batch_size_per_device_for_update=8 \
    worker.actor.micro_batch_size_per_device_for_experience=32 \
    worker.actor.fsdp.enable_full_shard=False \
    worker.actor.offload.offload_params=False \
    worker.actor.offload.offload_optimizer=False \
    worker.actor.model.enable_gradient_checkpointing=True \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.rollout.n=8 \
    worker.ref.use_torch_compile=False \
    worker.ref.fsdp.enable_cpu_offload=False \
    worker.rollout.enforce_eager=False \
    worker.rollout.max_model_len=2048 \
    worker.rollout.tensor_parallel_size=1 \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${NGPUS} \
    trainer.total_epochs=1 \
    trainer.max_steps=200 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="cepo" \
    trainer.val_freq=5 \
    trainer.save_limit=2 \
    trainer.val_before_train=True 2>&1 | tee "logs/${LOGS_NAME}"