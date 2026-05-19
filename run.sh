# cd /data/fast0/users/ahmed_heakl/cepo/easyr1
# conda activate easyr1

bash experiments/geo/cepo.sh
bash experiments/geo/cepo_weird.sh
bash experiments/geo/cepo_init0.5_weird.sh
bash experiments/geo/cepo_init0.5.sh
bash experiments/geo/cepo_default.sh
# bash experiments/cepo_exp_200steps_best.sh
# bash experiments/cepo_exp_200steps_best_100warmup.sh
# bash experiments/cepo_exp_eps0.8.sh
# bash experiments/cepo_exp_100steps_best_50warmup.sh
bash experiments/geo/grpo.sh

# grpo 2k 
bash scripts/test_omar.sh /data/fast0/users/ahmed_heakl/cepo/easyr1/checkpoints/cepo/qwen3_vl_4b_geo_cepo_lora_gt_0.5eps_weird/global_step_50/actor/huggingface
bash scripts/test_omar.sh ahmedheakl/cepo_geo_2b_lora_sdpo
bash scripts/test_omar.sh Qwen/Qwen3-VL-4B-Instruct
bash scripts/test_omar.sh Qwen/Qwen3-VL-4B-Instruct
bash scripts/test_omar.sh Qwen/Qwen3-VL-4B-Instruct



qwen3_vl_4b_geo_cepo_lora_gt_0.5eps_0.5init_weird
qwen3_vl_4b_geo_cepo_lora_gt_0.5eps
qwen3_vl_4b_geo_cepo_lora_gt_0.5init_weird
qwen3_vl_4b_geo_cepo_lora_gt_0.5init
qwen3_vl_4b_geo_cepo_lora


python scripts/model_merger.py -d checkpoints/cepo/qwen3_vl_4b_geo_cepo_lora_gt_0.5eps_0.5init_weird/global_step_50/actor
python scripts/model_merger.py -d checkpoints/cepo/qwen3_vl_4b_geo_cepo_lora_gt_0.5eps/global_step_50/actor
python scripts/model_merger.py -d checkpoints/cepo/qwen3_vl_4b_geo_cepo_lora_gt_0.5init_weird/global_step_50/actor
python scripts/model_merger.py -d checkpoints/cepo/qwen3_vl_4b_geo_cepo_lora_gt_0.5init/global_step_50/actor
python scripts/model_merger.py -d checkpoints/cepo/qwen3_vl_4b_geo_cepo_lora/global_step_50/actor

ROOT=/data/fast0/users/ahmed_heakl/cepo/easyr1/checkpoints/cepo
bash scripts/test_omar.sh $ROOT/qwen3_vl_4b_geo_cepo_lora_gt_0.5eps_0.5init_weird/global_step_50/actor/huggingface
bash scripts/test_omar.sh $ROOT/qwen3_vl_4b_geo_cepo_lora_gt_0.5eps/global_step_50/actor/huggingface
bash scripts/test_omar.sh $ROOT/qwen3_vl_4b_geo_cepo_lora_gt_0.5init_weird/global_step_50/actor/huggingface
bash scripts/test_omar.sh $ROOT/qwen3_vl_4b_geo_cepo_lora_gt_0.5init/global_step_50/actor/huggingface
bash scripts/test_omar.sh $ROOT/qwen3_vl_4b_geo_cepo_lora/global_step_50/actor/huggingface