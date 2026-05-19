# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface.
"""

import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any, Optional, Type

import numpy as np
import ray
import torch
from ray.experimental.tqdm_ray import tqdm
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from ..single_controller.base import Worker
from ..single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from ..single_controller.ray.base import create_colocated_worker_cls
from ..utils import torch_functional as VF
from ..utils.checkpoint import CHECKPOINT_TRACKER, find_latest_ckpt, remove_obsolete_ckpt
from ..utils.logger import Tracker
from ..utils.py_functional import convert_dict_to_str, timer, unflatten_dict
from ..utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from ..workers.fsdp_workers import FSDPWorker
from ..workers.reward import AutoRewardManager
from .config import PPOConfig
from .core_algos import (
    AdvantageEstimator,
    FixedKLController,
    KLController,
    apply_rlsd_reweighting,
    apply_cepo_weighting,
    compute_advantage_return,
    compute_kl,
    get_kl_controller,
)
from .metrics import (
    compute_data_metrics,
    compute_length_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)


class Role(IntEnum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = auto()
    Rollout = auto()
    ActorRollout = auto()
    Critic = auto()
    RefPolicy = auto()
    RewardModel = auto()
    ActorRolloutRef = auto()


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create ray resource pools for distributed training."""
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for different models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker."""
        return self.resource_pool_dict[self.mapping[role]]

    def get_num_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        gpus_available = ray.available_resources().get("GPU", 0)
        gpus_required = self.get_num_gpus()
        if gpus_available < gpus_required:
            raise ValueError(f"Total available GPUs {gpus_available} is less than total desired GPUs {gpus_required}.")


def apply_kl_penalty(data: DataProto, kl_ctrl: KLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards."""
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    response_mask = data.batch["response_mask"]

    # compute kl between ref_policy and current policy
    kld = compute_kl(data.batch["old_log_probs"], data.batch["ref_log_probs"], kl_penalty=kl_penalty)
    kld = kld * response_mask  # (batch_size, response_length)

    data.batch["token_level_rewards"] = token_level_scores - kl_ctrl.kl_coef * kld

    current_kl = torch.mean(VF.masked_mean(kld, mask=response_mask, dim=-1)).item()
    metrics = {"actor/kl_penalty": current_kl, "actor/kl_coef": kl_ctrl.kl_coef}

    # According to https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L880
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    return data, metrics


def compute_advantage(data: DataProto, adv_estimator: AdvantageEstimator, gamma: float = 1.0, lam: float = 1.0):
    """Compute advantage estimates for policy optimization."""
    adv_inputs = {
        "token_level_rewards": data.batch["token_level_rewards"],
        "response_mask": data.batch["response_mask"],
        "index": data.non_tensor_batch["uid"],
        "gamma": gamma,
        "lam": lam,
    }
    if "values" in data.batch:
        adv_inputs["values"] = data.batch["values"]

    if "reward_baselines" in data.batch:
        adv_inputs["reward_baselines"] = data.batch["reward_baselines"]

    advantages, returns = compute_advantage_return(adv_estimator, **adv_inputs)
    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def __init__(
        self,
        config: PPOConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        train_dataloader: StatefulDataLoader,
        val_dataloader: StatefulDataLoader,
        role_worker_mapping: dict[Role, Type[Worker]],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: Type[RayWorkerGroup] = RayWorkerGroup,
        reward_fn: Optional[AutoRewardManager] = None,
        val_reward_fn: Optional[AutoRewardManager] = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.val_reward_score = 0.0
        self.best_val_reward_score = -1.0
        self.best_global_step = None

        self.hybrid_engine = config.worker.hybrid_engine
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reward_model = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if config.algorithm.disable_kl:
            self.use_reference_policy = False
            self.kl_ctrl = FixedKLController(init_kl_coef=0.0)
            print("KL is disabled, no KL metrics will be logged. Please set `kl_coef=0` to log KL metrics.")
        else:
            self.use_reference_policy = True
            self.kl_ctrl = get_kl_controller(config.algorithm)

        if config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        else:
            self.use_critic = False

        if config.algorithm.adv_estimator not in list(AdvantageEstimator):
            raise NotImplementedError(f"Unknown advantage estimator: {config.algorithm.adv_estimator}.")

        if config.data.rollout_batch_size % config.worker.actor.global_batch_size != 0:
            raise ValueError("Rollout batch size must be divisible by actor global batch size.")

        if (
            config.data.rollout_batch_size * config.worker.rollout.n
        ) % config.worker.actor.micro_batch_size_per_device_for_experience != 0:
            raise ValueError(
                "Rollout batch size * rollout.n must be divisible by actor micro batch size for experience."
            )

        if self.use_critic:
            if config.data.rollout_batch_size % config.worker.critic.global_batch_size != 0:
                raise ValueError("Rollout batch size must be divisible by critic global batch size.")

            if (
                config.data.rollout_batch_size * config.worker.rollout.n
            ) % config.worker.critic.micro_batch_size_per_device_for_experience != 0:
                raise ValueError(
                    "Rollout batch size * rollout.n must be divisible by critic micro batch size for experience."
                )

        if (
            config.algorithm.adv_estimator in (AdvantageEstimator.GRPO, AdvantageEstimator.RLOO)
            and config.worker.rollout.n == 1
        ):
            raise ValueError("GRPO and RLOO algorithm need `config.worker.rollout.n > 1`.")

        if config.trainer.max_steps is not None:
            self.training_steps = config.trainer.max_steps
        elif config.data.mini_rollout_batch_size is not None:
            num_examples = len(train_dataloader) * config.data.mini_rollout_batch_size
            self.training_steps = num_examples // config.data.rollout_batch_size * config.trainer.total_epochs
        else:
            self.training_steps = len(train_dataloader) * config.trainer.total_epochs

        config.worker.actor.optim.training_steps = self.training_steps
        config.worker.critic.optim.training_steps = self.training_steps
        print(f"Total training steps: {self.training_steps}")

    def init_workers(self) -> None:
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor, rollout and ref
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRolloutRef)
            actor_rollout_ref_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRolloutRef], config=self.config.worker, role="actor_rollout_ref"
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout_ref"] = actor_rollout_ref_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic], config=self.config.worker, role="critic"
            )
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create a reward model if reward_fn is None
        if self.use_reward_model:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.RewardModel], config=self.config.worker, role="reward"
            )
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg: dict[str, FSDPWorker] = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reward_model:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_ref_wg = all_wg["actor_rollout_ref"]
        self.actor_rollout_ref_wg.init_model()

    def _save_checkpoint(self) -> None:
        # path: {save_checkpoint_path}/global_step_{global_step}/{actor,critic}
        if self.val_reward_score > self.best_val_reward_score:
            self.best_val_reward_score = self.val_reward_score
            self.best_global_step = self.global_step

        remove_obsolete_ckpt(
            self.config.trainer.save_checkpoint_path,
            self.global_step,
            self.best_global_step,
            self.config.trainer.save_limit,
        )
        folder_path = os.path.join(self.config.trainer.save_checkpoint_path, f"global_step_{self.global_step}")
        actor_path = os.path.join(folder_path, "actor")
        self.actor_rollout_ref_wg.save_checkpoint(actor_path, save_model_only=self.config.trainer.save_model_only)

        if self.use_critic:
            critic_path = os.path.join(folder_path, "critic")
            self.critic_wg.save_checkpoint(critic_path, save_model_only=self.config.trainer.save_model_only)

        dataloader_path = os.path.join(folder_path, "dataloader.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_path)

        checkpointer_tracker_info = {
            "best_global_step": self.best_global_step,
            "best_val_reward_score": round(self.best_val_reward_score, 4),
            "last_global_step": self.global_step,
            "last_actor_path": os.path.abspath(actor_path),
        }
        checkpointer_tracker_path = os.path.join(self.config.trainer.save_checkpoint_path, CHECKPOINT_TRACKER)
        with open(checkpointer_tracker_path, "w") as f:
            json.dump(checkpointer_tracker_info, f, ensure_ascii=False, indent=2)

    def _load_checkpoint(self) -> None:
        if self.config.trainer.load_checkpoint_path is not None:
            load_checkpoint_path = self.config.trainer.load_checkpoint_path
        elif self.config.trainer.find_last_checkpoint:
            load_checkpoint_path, tracker_info = find_latest_ckpt(self.config.trainer.save_checkpoint_path)
            if tracker_info is not None:
                self.best_val_reward_score = tracker_info.get("best_val_reward_score", 0.0)
                self.best_global_step = tracker_info.get("best_global_step", 0)
        else:
            load_checkpoint_path = None

        if load_checkpoint_path is None:
            return

        if "global_step_" not in load_checkpoint_path.strip(os.path.sep).split(os.path.sep)[-1]:
            raise ValueError("`load_checkpoint_path` should end with `global_step_*`.")

        print(f"Load from checkpoint: {load_checkpoint_path}.")
        self.global_step = int(load_checkpoint_path.strip(os.path.sep).split("global_step_")[-1])
        actor_path = os.path.join(load_checkpoint_path, "actor")
        self.actor_rollout_ref_wg.load_checkpoint(actor_path)
        if self.use_critic:
            critic_path = os.path.join(load_checkpoint_path, "critic")
            self.critic_wg.load_checkpoint(critic_path)

        dataloader_path = os.path.join(load_checkpoint_path, "dataloader.pt")
        if os.path.exists(dataloader_path):
            dataloader_state_dict = torch.load(dataloader_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"No dataloader state found at {dataloader_path}, will start from scratch.")

    def _maybe_log_val_generations(
        self, inputs: list[str], outputs: list[str], labels: list[str], scores: list[float]
    ) -> None:
        """Log a table of validation samples"""
        if self.config.trainer.val_generations_to_log <= 0:
            return

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, labels, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        samples = samples[: self.config.trainer.val_generations_to_log]
        self.logger.log_generation(samples, self.global_step)

    def _validate(self) -> dict[str, Any]:
        reward_tensor_lst = []
        # Lists to collect samples for the table
        sample_inputs, sample_outputs, sample_labels, sample_scores = [], [], [], []
        reward_metrics_lst = defaultdict(list)
        length_metrics_lst = defaultdict(list)
        print("Start validation...")
        self.actor_rollout_ref_wg.prepare_rollout_engine()
        for batch_dict in self.val_dataloader:
            test_batch = DataProto.from_single_dict(batch_dict)
            test_gen_batch = test_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
            )
            repeat_times = self.config.worker.rollout.val_override_config.get("n", 1)
            test_gen_batch.meta_info = self.config.worker.rollout.val_override_config
            test_gen_batch.meta_info["min_pixels"] = self.config.data.min_pixels
            test_gen_batch.meta_info["max_pixels"] = self.config.data.max_pixels
            test_gen_batch.meta_info["video_fps"] = self.config.data.video_fps

            test_gen_batch, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_ref_wg.world_size)
            test_output_gen_batch = self.actor_rollout_ref_wg.generate_sequences(test_gen_batch)
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch, pad_size=pad_size * repeat_times)

            # repeat to align with repeated responses in rollout
            test_batch = test_batch.repeat(repeat_times=repeat_times, interleave=True)
            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            reward_tensor, reward_metrics = ray.get(self.val_reward_fn.compute_reward.remote(test_batch))

            # store generations
            input_ids = test_batch.batch["prompts"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            output_ids = test_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_inputs.extend(input_texts)
            sample_outputs.extend(output_texts)
            sample_labels.extend(test_batch.non_tensor_batch["ground_truth"].tolist())
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            for key, value in reward_metrics.items():
                reward_metrics_lst[key].extend(value)

            for key, value in compute_length_metrics(test_batch).items():
                length_metrics_lst[key].append(value)

        self.actor_rollout_ref_wg.release_rollout_engine()
        self._maybe_log_val_generations(sample_inputs, sample_outputs, sample_labels, sample_scores)
        self.val_reward_score = torch.cat(reward_tensor_lst, dim=0).sum(-1).mean().item()
        val_reward_metrics = {f"val/{key}_reward": value for key, value in reduce_metrics(reward_metrics_lst).items()}
        val_length_metrics = {f"val_{key}": value for key, value in reduce_metrics(length_metrics_lst).items()}
        print("Finish validation.")
        return {"val/reward_score": self.val_reward_score, **val_reward_metrics, **val_length_metrics}

    def _balance_batch(self, batch: DataProto, metrics: dict[str, Any], logging_prefix: str = "global_seqlen") -> None:
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_ref_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _make_batch_data(self, metrics: dict[str, Any]) -> DataProto:
        batch = None
        all_metrics = defaultdict(list)
        num_try_make_batch = 0
        print("Start generating batch...")
        while True:
            num_try_make_batch += 1
            try:
                batch_dict = next(self.data_iterator)
            except StopIteration:
                self.data_iterator = iter(self.train_dataloader)
                batch_dict = next(self.data_iterator)

            meta_info = {
                "min_pixels": self.config.data.min_pixels,
                "max_pixels": self.config.data.max_pixels,
                "video_fps": self.config.data.video_fps,
            }
            new_batch: DataProto = DataProto.from_single_dict(batch_dict, meta_info=meta_info)
            new_batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
            )

            # pop those keys for generation
            gen_batch = new_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                meta_info_keys=["min_pixels", "max_pixels", "video_fps"],
            )

            # generate a batch
            gen_batch_output = self.actor_rollout_ref_wg.generate_sequences(gen_batch)

            if self.config.algorithm.adv_estimator == "remax":
                gen_baseline_batch = deepcopy(gen_batch)
                gen_baseline_batch.meta_info["temperature"] = 0
                gen_baseline_batch.meta_info["n"] = 1
                gen_baseline_output = self.actor_rollout_ref_wg.generate_sequences(gen_baseline_batch)

                new_batch = new_batch.union(gen_baseline_output)
                reward_baseline_tensor, _ = ray.get(self.reward_fn.compute_reward.remote(new_batch))
                reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                new_batch.batch["reward_baselines"] = reward_baseline_tensor
                del gen_baseline_batch, gen_baseline_output

            # repeat to align with repeated responses in rollout
            new_batch = new_batch.repeat(repeat_times=self.config.worker.rollout.n, interleave=True)
            new_batch = new_batch.union(gen_batch_output)

            # filter group
            if self.config.algorithm.online_filtering:
                reward_tensor, reward_metrics = ray.get(self.reward_fn.compute_reward.remote(new_batch))
                new_batch.batch["token_level_scores"] = reward_tensor
                for k, v in reward_metrics.items():
                    all_metrics[k].extend(v)

                filter_scores = reward_metrics[self.config.algorithm.filter_key]
                uids = new_batch.non_tensor_batch["uid"]
                uid2scores = defaultdict(list)
                for uid, score in zip(uids, filter_scores):
                    uid2scores[uid].append(score)

                uid2mean = {uid: np.mean(scores) for uid, scores in uid2scores.items()}
                kept_uids = [
                    uid
                    for uid, avg_score in uid2mean.items()
                    if avg_score > self.config.algorithm.filter_low and avg_score < self.config.algorithm.filter_high
                ]
                kept_sample_idxs = [idx for idx, uid in enumerate(uids) if uid in kept_uids]
                if len(kept_sample_idxs) == 0:
                    raise RuntimeError("No sample is kept after filtering. Please check your data.")

                new_batch = new_batch[kept_sample_idxs]

            batch = DataProto.concat([batch, new_batch]) if batch is not None else new_batch
            current_batch_size = len(batch) // self.config.worker.rollout.n
            rollout_batch_size = self.config.data.rollout_batch_size
            if current_batch_size < rollout_batch_size:
                print(f"{current_batch_size=} < {rollout_batch_size=}")
                max_try_make_batch = self.config.trainer.max_try_make_batch
                if max_try_make_batch <= 0 or num_try_make_batch < max_try_make_batch:
                    print(f"{num_try_make_batch=}. Continue generating...")
                else:
                    raise RuntimeError(
                        f"{num_try_make_batch=} >= {max_try_make_batch=}. Generated too many. Please check your data."
                    )
            else:
                print(f"{current_batch_size=} >= {rollout_batch_size=}. Finish generating.")
                if self.config.algorithm.online_filtering:
                    metrics.update({f"reward/{k}": v for k, v in reduce_metrics(all_metrics).items()})

                return batch[: self.config.data.rollout_batch_size * self.config.worker.rollout.n]
    def _build_cepo_teacher_batch(self, batch: DataProto) -> DataProto:
        from collections import defaultdict

        input_ids      = batch.batch["input_ids"]
        attention_mask = batch.batch["attention_mask"]
        position_ids   = batch.batch["position_ids"]
        responses      = batch.batch["responses"]
        scores         = batch.batch["token_level_scores"].sum(dim=-1)   # (bs,)
        ground_truth   = batch.non_tensor_batch["ground_truth"]
        index          = batch.non_tensor_batch["uid"]                   # ← FIX: string array

        bs         = input_ids.shape[0]
        full_len   = input_ids.shape[1]
        resp_len   = responses.shape[1]
        prompt_len = full_len - resp_len
        mrope      = position_ids.dim() == 3

        pad_id = self.tokenizer.pad_token_id or 0
        use_cot = getattr(self.config.algorithm, "cepo_use_cot_teacher", False)
        cot_k   = getattr(self.config.algorithm, "cepo_cot_prefix_len", 200)

        # ── identify groups ────────────────────────────────────────────────
        id2indices = defaultdict(list)
        for i in range(bs):
            id2indices[index[i]].append(i)

        # ── r^- for each group: worst rejected rollout ─────────────────────
        group2neg_idx   = {}
        group_all_correct = set()
        for gid, idxs in id2indices.items():
            rejected = [i for i in idxs if scores[i].item() < 0.5]
            if rejected:
                worst = min(rejected, key=lambda i: scores[i].item())
                group2neg_idx[gid] = worst
            else:
                group2neg_idx[gid] = None
                group_all_correct.add(gid)

        # ── r^+ for each group: best correct rollout (CoT mode only) ──────
        group2pos_idx = {}
        if use_cot:
            for gid, idxs in id2indices.items():
                correct = [i for i in idxs if scores[i].item() >= 0.5]
                if correct:
                    best = max(correct, key=lambda i: scores[i].item())
                    group2pos_idx[gid] = best
                else:
                    group2pos_idx[gid] = None

        # ── helper: first k non-pad tokens from a response row ─────────────
        def _response_prefix(idx, k):
            resp = responses[idx]
            pad_positions = (resp == pad_id).nonzero(as_tuple=True)[0]
            actual_len = pad_positions[0].item() if len(pad_positions) > 0 else resp.shape[0]
            return resp[:min(k, actual_len)]
        
        def _response_suffix(idx, k):
            resp = responses[idx]
            pad_positions = (resp == pad_id).nonzero(as_tuple=True)[0]
            actual_len = pad_positions[0].item() if len(pad_positions) > 0 else resp.shape[0]
            return resp[actual_len - k : actual_len] if use_cot else resp

        # ── build teacher sequences ────────────────────────────────────────
        pos_seqs, neg_seqs = [], []
        pos_masks, neg_masks = [], []
        pos_lens,  neg_lens  = [], []
        is_prefix = os.environ.get("CEPO_IS_PREFIX", "1") == "1"
        for i in range(bs):
            prompt_part   = input_ids[i, :prompt_len]
            response_part = responses[i]
            n_pad = int((attention_mask[i, :prompt_len] == 0).sum().item())
            real_prompt = prompt_part[n_pad:]

            # ── positive reference r^+ ─────────────────────────────────────
            if use_cot and group2pos_idx.get(index[i]) is not None and is_prefix:
                r_pos = _response_prefix(group2pos_idx[index[i]], cot_k)
            elif use_cot and group2pos_idx.get(index[i]) is not None and not is_prefix:
                r_pos = _response_suffix(group2pos_idx[index[i]], cot_k)
            else:
                gt_str = f" {str(ground_truth[i])}"
                r_pos_ids = self.tokenizer.encode(gt_str, add_special_tokens=False)
                r_pos = torch.tensor(r_pos_ids, dtype=torch.long)

            # ── negative reference r^- ─────────────────────────────────────
            neg_idx = group2neg_idx[index[i]]
            if neg_idx is None:
                r_neg = r_pos
            elif use_cot and is_prefix:
                r_neg = _response_prefix(neg_idx, cot_k)
            elif use_cot and not is_prefix:
                r_neg = _response_suffix(neg_idx, cot_k)
            else:
                neg_full_str = self.tokenizer.decode(
                    responses[neg_idx], skip_special_tokens=True
                )
                from mathruler.grader import extract_boxed_content
                neg_answer_str = extract_boxed_content(neg_full_str) or neg_full_str[:50]
                neg_answer_str = f" {neg_answer_str}"
                if not neg_answer_str.strip():
                    neg_answer_str = f" {str(ground_truth[i])}"
                r_neg = torch.tensor(
                    self.tokenizer.encode(neg_answer_str, add_special_tokens=False),
                    dtype=torch.long,
                )

            pos_core = torch.cat([real_prompt, r_pos, response_part])
            neg_core = torch.cat([real_prompt, r_neg, response_part])

            pos_seqs.append(pos_core); pos_lens.append(len(pos_core))
            neg_seqs.append(neg_core); neg_lens.append(len(neg_core))


        max_pos_len = max(pos_lens)
        max_neg_len = max(neg_lens)

        def left_pad_and_mask(seqs, max_len):
            ids_out  = torch.full((bs, max_len), pad_id, dtype=torch.long)
            mask_out = torch.zeros(bs, max_len, dtype=torch.long)
            for i, seq in enumerate(seqs):
                pad_size = max_len - len(seq)
                ids_out[i, pad_size:]  = seq
                mask_out[i, pad_size:] = 1
            return ids_out, mask_out

        pos_ids_t, pos_mask_t = left_pad_and_mask(pos_seqs, max_pos_len)
        neg_ids_t, neg_mask_t = left_pad_and_mask(neg_seqs, max_neg_len)
        for _vtok in [151655, 151656]:
            pos_ids_t[pos_ids_t == _vtok] = pad_id
            neg_ids_t[neg_ids_t == _vtok] = pad_id

        # ── position_ids for teacher sequences ─────────────────────────────
        def make_pos_ids(mask_tensor):
            """Build 1-D sequential position IDs aligned to non-pad tokens."""
            bs_, seqlen = mask_tensor.shape
            pos = torch.zeros(bs_, seqlen, dtype=torch.long)
            for i in range(bs_):
                n_pad_i = int((mask_tensor[i] == 0).sum().item())
                pos[i, n_pad_i:] = torch.arange(seqlen - n_pad_i, dtype=torch.long)
            return pos

        if not mrope:
            pos_posids = make_pos_ids(pos_mask_t)   # (bs, max_pos_len)
            neg_posids = make_pos_ids(neg_mask_t)   # (bs, max_neg_len)
        else:
            # For teacher passes: simple sequential positions (all components identical).
            # Critical: after unpadding, each sample's first token has position 0,
            # so prepare_fa2_from_position_ids correctly detects sequence boundaries.
            # Spatial accuracy doesn't matter — no pixel_values are passed to teacher.
            n_comp = position_ids.shape[1]  # e.g. 3 for Qwen3-VL

            def make_mrope_seq_pos(mask_tensor, n_comp):
                bs_, seqlen = mask_tensor.shape
                pos = torch.zeros(bs_, n_comp, seqlen, dtype=torch.long)
                for i in range(bs_):
                    n_pad_i = int((mask_tensor[i] == 0).sum().item())
                    seq_pos = torch.arange(seqlen - n_pad_i, dtype=torch.long)
                    for c in range(n_comp):
                        pos[i, c, n_pad_i:] = seq_pos
                return pos

            pos_posids = make_mrope_seq_pos(pos_mask_t, n_comp)  # (bs, n_comp, max_pos_len)
            neg_posids = make_mrope_seq_pos(neg_mask_t, n_comp)  # (bs, n_comp, max_neg_len)
        
        
        teacher_batch = DataProto.from_dict(
            tensors={
                "pos_input_ids":      pos_ids_t,
                "pos_attention_mask": pos_mask_t,
                "pos_position_ids":   pos_posids,
                "neg_input_ids":      neg_ids_t,
                "neg_attention_mask": neg_mask_t,
                "neg_position_ids":   neg_posids,
                "responses":          responses,
            },
            non_tensors={
                "uid": batch.non_tensor_batch.get("uid", np.array([""] * bs, dtype=object)),
                **(
                    {"multi_modal_data": batch.non_tensor_batch["multi_modal_data"]}
                    if "multi_modal_data" in batch.non_tensor_batch
                    else {}
                ),
            },
            meta_info={"temperature": self.config.worker.rollout.temperature},
        )
        return teacher_batch
    
    def _build_teacher_inputs(self, batch: DataProto) -> DataProto:
        """Build teacher-mode inputs for RLSD.

        Student:  [left_pad | prompt_tokens               | response_tokens]
        Teacher:  [left_pad | prompt_tokens | answer_tokens | response_tokens]

        Critical fix: use attention_mask[i].argmax() to compute left_pad (number
        of leading padding tokens). DO NOT use (mask == 0).sum() — that counts
        BOTH leading AND trailing post-EOS zeros, causing the teacher sequence to
        silently drop image tokens from the prompt, producing a feature/token
        count mismatch in _get_input_embeds.
        """
        input_ids      = batch.batch["input_ids"]        # (bs, full_len)
        attention_mask = batch.batch["attention_mask"]    # (bs, full_len)
        position_ids   = batch.batch["position_ids"]      # (bs, L) or (bs, C, L)
        responses      = batch.batch["responses"]          # (bs, response_len)

        bs, full_len  = input_ids.shape
        response_len  = responses.size(1)
        device        = input_ids.device
        is_mrope      = (position_ids.dim() == 3)
        n_components  = position_ids.size(1) if is_mrope else None

        answers = batch.non_tensor_batch.get("answer", [""] * bs)

        answer_texts    = [f" {str(a)}" for a in answers]
        answer_encoding = self.tokenizer(
            answer_texts,
            add_special_tokens=False,
            padding=False,
            return_tensors=None,
        )

        pad_token_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )

        teacher_input_ids_list    = []
        teacher_attn_mask_list    = []
        teacher_position_ids_list = []

        for i in range(bs):
            # ── FIX: argmax finds the first 1 = number of LEADING zeros only.
            # (attention_mask[i] == 0).sum() is WRONG — it also counts trailing
            # post-EOS zeros in the response, causing image tokens to be skipped.
            left_pad   = int(attention_mask[i].argmax().item())
            prompt_end = full_len - response_len   # always max_prompt_length

            prompt_tokens   = input_ids[i, left_pad:prompt_end]
            response_tokens = responses[i]
            answer_ids      = torch.tensor(
                answer_encoding["input_ids"][i],
                dtype=torch.long,
                device=device,
            )
            ans_len           = answer_ids.size(0)
            actual_prompt_len = prompt_end - left_pad

            # ── Token sequence ────────────────────────────────────────────────
            tok_parts = [prompt_tokens]
            if ans_len > 0:
                tok_parts.append(answer_ids)
            tok_parts.append(response_tokens)

            teacher_seq  = torch.cat(tok_parts)
            teacher_mask = torch.ones(teacher_seq.size(0), dtype=torch.long, device=device)

            # ── Position IDs ──────────────────────────────────────────────────
            if is_mrope:
                prompt_pos = position_ids[i, :, left_pad:prompt_end]  # (C, actual_prompt_len)

                if actual_prompt_len > 0:
                    last_prompt_pos = prompt_pos[:, -1:]               # (C, 1)
                else:
                    last_prompt_pos = torch.zeros(
                        n_components, 1, dtype=position_ids.dtype, device=device
                    )

                pos_parts = [prompt_pos]

                if ans_len > 0:
                    ans_offsets          = torch.arange(1, ans_len + 1, device=device).unsqueeze(0)
                    ans_pos              = last_prompt_pos + ans_offsets   # (C, ans_len)
                    last_pos_before_resp = ans_pos[:, -1:]                 # (C, 1)
                    pos_parts.append(ans_pos)
                else:
                    last_pos_before_resp = last_prompt_pos

                resp_offsets = torch.arange(1, response_len + 1, device=device).unsqueeze(0)
                resp_pos     = last_pos_before_resp + resp_offsets         # (C, response_len)
                pos_parts.append(resp_pos)

                teacher_pos = torch.cat(pos_parts, dim=1)                  # (C, teacher_len)

            else:
                teacher_pos = torch.arange(
                    teacher_seq.size(0), dtype=torch.long, device=device
                )

            teacher_input_ids_list.append(teacher_seq)
            teacher_attn_mask_list.append(teacher_mask)
            teacher_position_ids_list.append(teacher_pos)

        # ── Left-pad to uniform length within the batch ───────────────────────
        max_teacher_len = max(s.size(0) for s in teacher_input_ids_list)

        padded_input_ids    = []
        padded_attn_mask    = []
        padded_position_ids = []

        for seq, mask, pos in zip(
            teacher_input_ids_list, teacher_attn_mask_list, teacher_position_ids_list
        ):
            pad_len = max_teacher_len - seq.size(0)

            padded_input_ids.append(torch.cat([
                torch.full((pad_len,), pad_token_id, dtype=torch.long, device=device),
                seq,
            ]))
            padded_attn_mask.append(torch.cat([
                torch.zeros(pad_len, dtype=torch.long, device=device),
                mask,
            ]))
            if is_mrope:
                padded_position_ids.append(torch.cat([
                    torch.zeros(n_components, pad_len, dtype=pos.dtype, device=device),
                    pos,
                ], dim=1))
            else:
                padded_position_ids.append(torch.cat([
                    torch.zeros(pad_len, dtype=pos.dtype, device=device),
                    pos,
                ]))

        batch.batch["teacher_input_ids"]      = torch.stack(padded_input_ids)
        batch.batch["teacher_attention_mask"] = torch.stack(padded_attn_mask)
        batch.batch["teacher_position_ids"]   = torch.stack(padded_position_ids)

        return batch


    def _compute_rlsd_lambda(self) -> float:
        """Linearly decay λ from rlsd_lambda_init → 0 over rlsd_lambda_decay_steps.

        After decay_steps the lambda is fixed at 0, meaning RLSD becomes
        identical to standard GRPO for the remainder of training.
        This two-phase structure (dense credit → standard RLVR) is key to
        RLSD's stability advantage over OPSD.
        """
        cfg  = self.config.algorithm
        step = self.global_step
        if step >= cfg.rlsd_lambda_decay_steps:
            return 0.0
        return cfg.rlsd_lambda_init * (1.0 - step / cfg.rlsd_lambda_decay_steps)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        self.logger = Tracker(loggers=self.config.trainer.logger, config=self.config.to_dict())
        self.global_step = 0
        self._cepo_lambda = self.config.algorithm.cepo_lambda_init if self.config.algorithm.use_cepo else 0.0
        main_tqdm = tqdm(range(self.training_steps), desc="Running step", position=0)
        val_metrics: Optional[dict[str, Any]] = None

        # load checkpoint before doing anything
        self._load_checkpoint()
        main_tqdm.update(self.global_step)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.val_before_train:
            val_metrics = self._validate()
            self.logger.log(data=val_metrics, step=self.global_step)
            if self.config.trainer.val_only:
                return

        self.data_iterator = iter(self.train_dataloader)
        while self.global_step < self.training_steps:
            self.global_step += 1

            metrics, timing_raw = {}, {}
            with timer("step", timing_raw):
                # make a batch of data
                with timer("gen", timing_raw):
                    self.actor_rollout_ref_wg.prepare_rollout_engine()
                    batch = self._make_batch_data(metrics=metrics)
                    self.actor_rollout_ref_wg.release_rollout_engine()

                # balance the number of valid tokens on each dp rank.
                # NOTE: this breaks the order of data inside the batch.
                # Please take care when you implement group based adv computation such as GRPO and rloo
                self._balance_batch(batch, metrics=metrics)

                # compute global valid tokens
                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                # compute reward
                if "token_level_scores" not in batch.batch:
                    with timer("reward", timing_raw):
                        reward_ref = self.reward_fn.compute_reward.remote(batch)

                # recompute old_log_probs
                with timer("old", timing_raw):
                    old_log_probs = self.actor_rollout_ref_wg.compute_log_probs(batch)
                    batch = batch.union(old_log_probs)
                    
                # ── RLSD: compute teacher log probs (one extra forward pass) ──
                if self.config.algorithm.adv_estimator == AdvantageEstimator.RLSD:
                    with timer("teacher", timing_raw):
                        batch = self._build_teacher_inputs(batch)
                        teacher_log_probs_out = self.actor_rollout_ref_wg.compute_teacher_log_probs(batch)
                        batch = batch.union(teacher_log_probs_out)

                # compute ref_log_probs
                if self.use_reference_policy:
                    with timer("ref", timing_raw):
                        ref_log_probs = self.actor_rollout_ref_wg.compute_ref_log_probs(batch)
                        batch = batch.union(ref_log_probs)

                # compute values
                if self.use_critic:
                    with timer("values", timing_raw):
                        values = self.critic_wg.compute_values(batch)
                        batch = batch.union(values)

                with timer("adv", timing_raw):
                    if "token_level_scores" not in batch.batch:
                        # get token level scores asynchronously
                        reward_tensor, reward_metrics = ray.get(reward_ref)
                        batch.batch["token_level_scores"] = reward_tensor
                        reward_metrics = {f"reward/{k}": v for k, v in reduce_metrics(reward_metrics).items()}
                        metrics.update(reward_metrics)

                    # apply kl penalty if available
                    if not self.config.algorithm.use_kl_loss and self.use_reference_policy:
                        # apply kl penalty to reward
                        batch, kl_metrics = apply_kl_penalty(batch, self.kl_ctrl, self.config.algorithm.kl_penalty)
                        metrics.update(kl_metrics)
                    else:
                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                    _adv_estimator = self.config.algorithm.adv_estimator
                    if _adv_estimator == AdvantageEstimator.RLSD:
                        _adv_estimator = AdvantageEstimator.GRPO  # base GRPO advantages
                    
                    # compute advantages, executed on the driver process
                    batch = compute_advantage(
                        batch,
                        adv_estimator=_adv_estimator,
                        gamma=self.config.algorithm.gamma,
                        lam=self.config.algorithm.lam,
                    )
                    # ── CEPO: contrastive token-level advantage reweighting ────────
                    if self.config.algorithm.use_cepo and self._cepo_lambda > 0:
                        with timer("cepo_teacher", timing_raw):
                            teacher_batch = self._build_cepo_teacher_batch(batch)
                            teacher_batch, teacher_pad = pad_dataproto_to_divisor(
                                teacher_batch, self.actor_rollout_ref_wg.world_size
                            )
                            teacher_output = self.actor_rollout_ref_wg.compute_cepo_teacher_log_probs(
                                teacher_batch
                            )
                            teacher_output = unpad_dataproto(teacher_output, pad_size=teacher_pad)

                        pos_lp = teacher_output.batch["pos_teacher_log_probs"]  # (bs, resp_len)
                        neg_lp = teacher_output.batch["neg_teacher_log_probs"]  # (bs, resp_len)

                        _adv_pre  = batch.batch["advantages"]                  # reference before CEPO
                        _mask     = batch.batch["response_mask"].bool()
                        _delta    = (pos_lp - neg_lp)
                        _sign_A   = torch.sign((_adv_pre * batch.batch["response_mask"]).sum(-1, keepdim=True))
                        
                        # ── Sign-conditioned masks (Probe A + C) ───────────────────
                        # _sign_A is (bs, 1); broadcast to (bs, resp_len) for token masking
                        _sign_A_tok = _sign_A.expand_as(_delta)          # (bs, resp_len)
                        _mask_pos = (_mask & (_sign_A_tok > 0))           # tokens in correct rollouts
                        _mask_neg = (_mask & (_sign_A_tok < 0))           # tokens in wrong rollouts


                        batch.batch["advantages"] = apply_cepo_weighting(
                            advantages=_adv_pre,
                            response_mask=batch.batch["response_mask"],
                            pos_teacher_log_probs=pos_lp,
                            neg_teacher_log_probs=neg_lp,
                            lam=self._cepo_lambda,
                            eps_w=self.config.algorithm.cepo_eps_w,
                        )

                        # Linearly decay lambda toward 0
                        if self.config.algorithm.cepo_lambda_schedule == "linear":
                            warmup = max(1, self.config.algorithm.cepo_warmup_steps)
                            self._cepo_lambda = max(
                                0.0,
                                self.config.algorithm.cepo_lambda_init * (1.0 - self.global_step / warmup),
                            )
                        # ── Scalar diagnostics ─────────────────────────────────────
                        _delta_mean_all = _delta[_mask].mean().item()
                        _delta_std_all  = _delta[_mask].std().item()

                        # Probe A + C: sign-conditioned means
                        _delta_mean_pos = _delta[_mask_pos].mean().item() if _mask_pos.any() else 0.0
                        _delta_mean_neg = _delta[_mask_neg].mean().item() if _mask_neg.any() else 0.0
                        _delta_std_pos  = _delta[_mask_pos].std().item()  if _mask_pos.any() else 0.0
                        _delta_std_neg  = _delta[_mask_neg].std().item()  if _mask_neg.any() else 0.0
                        # frac_wrong_dir per stratum
                        _fwd_pos = (_delta[_mask_pos] < 0).float().mean().item() if _mask_pos.any() else 0.5
                        _fwd_neg = (_delta[_mask_neg] > 0).float().mean().item() if _mask_neg.any() else 0.5

                        # ── Probe B: delta histogram dump (every 5 steps) ──────────
                        _dump_freq = getattr(self.config.algorithm, "cepo_histogram_freq", 5)
                        if self.global_step % _dump_freq == 0:
                            import os, json
                            _hist_dir = os.path.join(
                                self.config.trainer.save_checkpoint_path, "cepo_histograms"
                            )
                            os.makedirs(_hist_dir, exist_ok=True)
                            _hist_path = os.path.join(_hist_dir, f"step_{self.global_step:04d}.json")
                            # Detach and move to CPU — keep raw (unclipped) values
                            _delta_pos_vals = _delta[_mask_pos].detach().cpu().float().tolist()
                            _delta_neg_vals = _delta[_mask_neg].detach().cpu().float().tolist()
                            with open(_hist_path, "w") as _hf:
                                json.dump({
                                    "step": self.global_step,
                                    "delta_pos": _delta_pos_vals,   # delta from correct rollouts
                                    "delta_neg": _delta_neg_vals,   # delta from wrong rollouts
                                }, _hf)

                        # ── Log to wandb + print ───────────────────────────────────
                        metrics["cepo/lambda"]         = self._cepo_lambda
                        metrics["cepo/delta_mean"]     = _delta_mean_all
                        metrics["cepo/delta_std"]      = _delta_std_all
                        metrics["cepo/delta_mean_pos"] = _delta_mean_pos   # μ+ — want > 0
                        metrics["cepo/delta_mean_neg"] = _delta_mean_neg   # μ- — want < 0
                        metrics["cepo/delta_std_pos"]  = _delta_std_pos
                        metrics["cepo/delta_std_neg"]  = _delta_std_neg
                        metrics["cepo/frac_wrong_dir"] = (
                            (_delta * _sign_A_tok)[_mask] < 0
                        ).float().mean().item()
                        metrics["cepo/frac_wrong_dir_pos"] = _fwd_pos     # want < 0.2
                        metrics["cepo/frac_wrong_dir_neg"] = _fwd_neg     # want < 0.4
                        metrics["cepo/adv_pre"]        = _adv_pre.abs().mean().item()
                        metrics["cepo/adv_post"]       = batch.batch["advantages"].abs().mean().item()

                        print(
                            f"[CEPO] step={self.global_step}, lambda={self._cepo_lambda:.4f}, "
                            f"delta_mean={_delta_mean_all:.4f}, delta_std={_delta_std_all:.4f}, "
                            f"mu+={_delta_mean_pos:.4f}, mu-={_delta_mean_neg:.4f}, "
                            f"fwd={metrics['cepo/frac_wrong_dir']:.3f} "
                            f"(pos={_fwd_pos:.3f}, neg={_fwd_neg:.3f}), "
                            f"adv_pre={_adv_pre.abs().mean():.4f}, "
                            f"adv_post={batch.batch['advantages'].abs().mean():.4f}"
                        )
                    # ──────────────────────────────────────────────────────────────
                    
                    # ── RLSD: apply token-level credit reweighting ─────────
                    if self.config.algorithm.adv_estimator == AdvantageEstimator.RLSD:
                        lambda_coef = self._compute_rlsd_lambda()
                        rlsd_advantages, rlsd_metrics = apply_rlsd_reweighting(
                            advantages=batch.batch["advantages"],
                            old_log_probs=batch.batch["old_log_probs"],
                            teacher_log_probs=batch.batch["teacher_log_probs"],
                            response_mask=batch.batch["response_mask"],
                            lambda_coef=lambda_coef,
                            epsilon_w=self.config.algorithm.rlsd_epsilon_w,
                        )
                        batch.batch["advantages"] = rlsd_advantages
                        metrics.update(rlsd_metrics)

                # update critic
                if self.use_critic:
                    with timer("update_critic", timing_raw):
                        critic_output = self.critic_wg.update_critic(batch)

                    critic_metrics = reduce_metrics(critic_output.non_tensor_batch)
                    metrics.update(critic_metrics)

                # update actor
                if self.config.trainer.critic_warmup <= self.global_step:
                    with timer("update_actor", timing_raw):
                        actor_output = self.actor_rollout_ref_wg.update_actor(batch)

                    actor_metrics = reduce_metrics(actor_output.non_tensor_batch)
                    metrics.update(actor_metrics)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.val_freq > 0
                    and self.global_step % self.config.trainer.val_freq == 0
                ):
                    with timer("validation", timing_raw):
                        val_metrics = self._validate()

                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and self.global_step % self.config.trainer.save_freq == 0:
                    with timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()

            # collect metrics
            num_gpus = self.resource_pool_manager.get_num_gpus()
            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, num_gpus=num_gpus))

            self.logger.log(data=metrics, step=self.global_step)
            main_tqdm.update()

        # perform validation after training
        if self.val_reward_fn is not None:
            if (
                val_metrics is None
                or self.config.trainer.val_freq <= 0
                or self.global_step % self.config.trainer.val_freq != 0
            ):
                val_metrics = self._validate()
                self.logger.log(data=val_metrics, step=self.global_step)

            print(f"Final validation metrics:\n{convert_dict_to_str(unflatten_dict(val_metrics))}")

        if self.config.trainer.save_freq <= 0 or self.global_step % self.config.trainer.save_freq != 0:
            self._save_checkpoint()
