# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Dict, List, Optional, Type

import numpy as np
import ray
import torch
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.rollout.async_server import AsyncLLMServerManager
from gigpo import core_gigpo

from agent_system.multi_turn_rollout import TrajectoryCollector, adjust_batch

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"
    GRPO_PASSK = "grpo_passk"
    GiGPO = 'gigpo'


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0) for node, node_info in node_available_resources.items()}

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])
        if total_available_gpus < total_required_gpus:
            raise ValueError(f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}")

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}" + "cannot be satisfied in this ray cluster")


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl", multi_turn=False):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    if multi_turn:
        loss_mask = data.batch["loss_mask"]
        response_mask = loss_mask[:, -response_length:]
    else:
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty)  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics

def apply_ahsr(
    data: DataProto,
    lambda_plus:  float = 2.0,
    lambda_minus: float = 0.3,
    lambda_decay: float = 4.0,
    lambda_min:   float = 0.0,
):
    """Adaptive Asymmetric Hindsight Skill Reward (A-AHSR).

    Per task-group g, compute skill advantage:
        delta_g  = mean(R_skill_g) - mean(R_noskill_g)
        lambda+_g = max(lambda_min, lambda_plus - lambda_decay * max(0, delta_g))

    Per skill-arm trajectory i:
        delta_i   = R_i - mean(R_noskill_g)
        R_int_i   = lambda+_g * max(0, delta_i) - lambda_minus * max(0, -delta_i)
        R_tilde_i = R_i + R_int_i

    Early training (skill worse than no-skill, delta_g <= 0):
        lambda+_g = lambda_plus (maximum, strong encouragement)
    Late training (skill clearly better, delta_g >= lambda_plus/lambda_decay):
        lambda+_g = 0 (no amplification, natural convergence)
    """
    from collections import defaultdict
    wsm = data.non_tensor_batch.get('with_skills_mask')
    if wsm is None or not wsm.any() or wsm.all():
        return data, {}

    token_rewards = data.batch['token_level_rewards']
    if 'response_mask' not in data.batch:
        data.batch['response_mask'] = compute_response_mask(data)
    response_mask = data.batch['response_mask']
    uid      = data.non_tensor_batch['uid']
    traj_uid = data.non_tensor_batch['traj_uid']
    bsz      = token_rewards.shape[0]

    ep_rewards = token_rewards.sum(dim=-1)  # (bs,)

    # Collect per-group rewards for both arms, deduplicate by traj_uid
    noskill_per_group = defaultdict(list)
    skill_per_group   = defaultdict(list)
    seen = set()
    for i in range(bsz):
        key = (uid[i], traj_uid[i])
        if key in seen:
            continue
        seen.add(key)
        if bool(wsm[i]):
            skill_per_group[uid[i]].append(ep_rewards[i].item())
        else:
            noskill_per_group[uid[i]].append(ep_rewards[i].item())

    # Compute per-group baseline and adaptive lambda+
    baseline      = {g: sum(v)/len(v) for g, v in noskill_per_group.items() if v}
    skill_mean    = {g: sum(v)/len(v) for g, v in skill_per_group.items() if v}
    lambda_plus_g = {}
    for g in baseline:
        delta_g = skill_mean.get(g, baseline[g]) - baseline[g]
        lambda_plus_g[g] = max(lambda_min, lambda_plus - lambda_decay * max(0.0, delta_g))

    # Compute intrinsic reward per skill-arm trajectory
    intrinsic = torch.zeros(bsz, device=token_rewards.device)
    seen2 = set()
    for i in range(bsz):
        if not bool(wsm[i]):
            continue
        key = (uid[i], traj_uid[i])
        if key in seen2:
            continue
        seen2.add(key)
        g = uid[i]
        if g not in baseline:
            continue
        lp    = lambda_plus_g[g]
        mu    = baseline[g]
        delta = ep_rewards[i].item() - mu
        intrinsic[i] = lp * max(0.0, delta) - lambda_minus * max(0.0, -delta)

    # Apply at last valid token
    last_idx = (response_mask.long().cumsum(dim=-1) ==
                response_mask.long().sum(dim=-1, keepdim=True)).float().argmax(dim=-1)
    new_rewards = token_rewards.clone()
    for i in range(bsz):
        if intrinsic[i].item() != 0.0:
            new_rewards[i, last_idx[i]] = new_rewards[i, last_idx[i]] + intrinsic[i]
    data.batch['token_level_rewards'] = new_rewards

    # Metrics
    skill_idx  = torch.tensor([bool(w) for w in wsm], dtype=torch.bool)
    intr_skill = intrinsic[skill_idx]
    lp_vals    = [lambda_plus_g.get(uid[i], lambda_plus)
                  for i in range(bsz) if bool(wsm[i])]
    metrics = {
        'skill/ahsr_intrinsic_mean':   intr_skill.mean().item() if intr_skill.numel() > 0 else 0.0,
        'skill/ahsr_intrinsic_pos':    (intr_skill > 0).float().mean().item() if intr_skill.numel() > 0 else 0.0,
        'skill/ahsr_intrinsic_neg':    (intr_skill < 0).float().mean().item() if intr_skill.numel() > 0 else 0.0,
        'skill/ahsr_lambda_plus_mean': sum(lp_vals)/len(lp_vals) if lp_vals else lambda_plus,
    }
    return data, metrics


def apply_invalid_action_penalty(data: DataProto, invalid_action_penalty_coef=float):
    reward_tensor = data.batch['token_level_scores']
    if 'step_rewards' in data.batch.keys():
        step_rewards = data.batch['step_rewards']
    for i in range(len(data)):
        data_item = data[i]  # DataProtoItem

        prompt_ids = data_item.batch['prompts']

        prompt_length = prompt_ids.shape[-1]

        valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()

        action_valids = data_item.non_tensor_batch['is_action_valid'].astype(np.float32)
        action_invalids = torch.tensor(1 - action_valids, dtype=torch.float32, device=prompt_ids.device).squeeze(0)
        # invalid action penalty
        # assert reward_tensor[i, valid_response_length - 1] != 0.0, f'i={i}'
        reward_tensor[i, valid_response_length - 1] -= invalid_action_penalty_coef * action_invalids

        if 'step_rewards' in data.batch.keys():
            step_rewards[i] -= invalid_action_penalty_coef * action_invalids
    
    valid_action_ratio = np.mean(data.non_tensor_batch['is_action_valid'].astype(np.float32)).item()
    metrics = {'episode/valid_action_ratio': valid_action_ratio}
    return data, metrics

def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, multi_turn=False, norm_adv_by_std_in_grpo=True, step_advantage_w=1.0, gigpo_mode="mean_std_norm", gigpo_enable_similarity=False, gigpo_similarity_thresh=0.95, **kwargs):
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator: The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in GRPO. Defaults to True.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch:
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == AdvantageEstimator.GAE:
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if kwargs.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                kwargs.get("pf_ppo_reweight_method", "pow"),
                kwargs.get("pf_ppo_weight_pow", 2.0),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        grpo_calculation_mask = data.batch["response_mask"]
        if multi_turn:
            response_length = grpo_calculation_mask.size(1)
            grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            traj_index=data.non_tensor_batch['traj_uid'],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GRPO_PASSK:
        advantages, returns = core_algos.compute_grpo_passk_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            traj_index=data.non_tensor_batch['traj_uid'],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE:
        advantages, returns = core_algos.compute_reinforce_plus_plus_baseline_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            traj_index=data.non_tensor_batch['traj_uid'],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REMAX:
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            reward_baselines=data.batch["reward_baselines"],
            response_mask=data.batch["response_mask"],
        )

        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            traj_index=data.non_tensor_batch['traj_uid'],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GiGPO:
        advantages, returns = core_gigpo.compute_gigpo_outcome_advantage(
            token_level_rewards=data.batch['token_level_rewards'], # for episode group reward computing
            step_rewards=data.batch['step_rewards'], # for step group reward computing
            response_mask=data.batch['response_mask'],
            anchor_obs=data.non_tensor_batch['anchor_obs'],
            index=data.non_tensor_batch['uid'],
            traj_index=data.non_tensor_batch['traj_uid'],
            step_advantage_w=step_advantage_w,
            mode=gigpo_mode,
            enable_similarity=gigpo_enable_similarity,
            similarity_thresh=gigpo_similarity_thresh,
            )
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    """Context manager for timing code execution.

    This utility function measures the execution time of code within its context
    and accumulates the timing information in the provided dictionary.

    Args:
        name (str): The name/identifier for this timing measurement.
        timing_raw (Dict[str, float]): Dictionary to store timing information.

    Yields:
        None: This is a context manager that yields control back to the code block.
    """
    with Timer(name=name, logger=None) as timer:
        yield
    if name not in timing_raw:
        timing_raw[name] = 0
    timing_raw[name] += timer.last


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name="cuda",
        traj_collector: TrajectoryCollector = None,
        envs=None,
        val_envs=None,
    ):
        """Initialize distributed PPO trainer with Ray backend."""

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        self.envs = envs
        self.val_envs = val_envs
        self.traj_collector = traj_collector

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name
        self.validation_generations_logger = ValidationGenerationsLogger()

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get('lora_rank', 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.GRPO_PASSK,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
            AdvantageEstimator.GiGPO
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % n_gpus == 0, f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove '{name}.{param}' because only '*_{param_per_gpu}'" + "is supported (the former is deprecated).")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor",
            )

            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu, "critic")

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model")

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            # assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == "fsdp" and (config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1 or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1):
            assert config.actor_rollout_ref.model.use_remove_padding, "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == "fsdp":
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert config.critic.model.use_remove_padding, "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        if config.data.get("val_batch_size", None) is not None:
            print("WARNING: val_batch_size is deprecated." + " Validation datasets are sent to inference engines as a whole batch," + " which will schedule the memory themselves.")

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, "validation gen temperature should be greater than 0 when enabling do_sample"

        # check multi_turn with tool config
        if config.actor_rollout_ref.rollout.multi_turn.enable:
            assert config.actor_rollout_ref.rollout.multi_turn.tool_config_path is not None, "tool_config_path must be set when enabling multi_turn with tool, due to no role-playing support"
            assert config.algorithm.adv_estimator in [AdvantageEstimator.GRPO], "only GRPO is tested for multi-turn with tool"

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(self.config.data.train_files, self.config.data, self.tokenizer, self.processor)
        if val_dataset is None:
            val_dataset = create_rl_dataset(self.config.data.val_files, self.config.data, self.tokenizer, self.processor)
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: {len(self.val_dataloader)}")

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        with open(filename, "w") as f:
            for i in range(n):
                entry = {k: v[i] for k, v in base_data.items()}
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"Dumped generations to {filename}")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _validate(self):
        reward_tensor_lst = []
        data_source_lst = []
        tool_calling_list = []
        traj_uid_list = []
        success_rate_dict = {}

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True)

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
            if "multi_modal_data" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            if "env_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("env_kwargs")
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # # pad to be divisible by dp_size
            # test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            # test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)

            # # unpad
            # test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            ################ agent-environment loop ###############
            test_output_gen_batch = self.traj_collector.multi_turn_loop(
                                                    gen_batch=test_gen_batch,
                                                    actor_rollout_wg=self.actor_rollout_wg,
                                                    envs=self.val_envs,
                                                    is_train=False,
                                                    )
            print('validation generation end')
            del test_batch
            test_batch = test_output_gen_batch
            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            # test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))
            tool_calling_list.append(test_output_gen_batch.non_tensor_batch['tool_callings'])
            traj_uid_list.append(test_output_gen_batch.non_tensor_batch['traj_uid'])
            # success rate
            for k in test_batch.non_tensor_batch.keys():
                if 'success_rate' in k:
                    if k not in success_rate_dict:
                        success_rate_dict[k] = []
                    success_rate_dict[k].append(test_batch.non_tensor_batch[k][0])
                    # all success_rate should be the same
                    for i in range(1, len(test_batch.non_tensor_batch[k])):
                        assert test_batch.non_tensor_batch[k][0] == test_batch.non_tensor_batch[k][i], f'not all success_rate are the same, 0: {test_batch.non_tensor_batch[k][0]}, {i}: {test_batch.non_tensor_batch[k][i]}'

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)
        tool_callings = np.concatenate(tool_calling_list, axis=0)
        traj_uids = np.concatenate(traj_uid_list, axis=0)
        success_rate = {k: np.mean(v) for k, v in success_rate_dict.items()}

        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        # evaluate tool call based on data source
        # the values in tool_callings represent the tool call count for each trajectory; however, since the batch is expanded by step, we only need to take one value for each unique trajectories.
        data_source_tool_calling = {}
        unique_traj_uid, unique_idx = np.unique(traj_uids, return_index=True)
        unique_data_sources = data_sources[unique_idx]
        unique_tool_callings = tool_callings[unique_idx]

        for i in range(unique_tool_callings.shape[0]):
            data_source = unique_data_sources[i]
            if data_source not in data_source_tool_calling:
                data_source_tool_calling[data_source] = []
            data_source_tool_calling[data_source].append(unique_tool_callings[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val/{data_source}/test_score'] = np.mean(rewards)

        for data_source, tool_calls in data_source_tool_calling.items():
            metric_dict[f'val/{data_source}/tool_call_count/mean'] = np.mean(tool_calls)
            # metric_dict[f'val/{data_source}/tool_call_count/max'] = np.max(tool_calls)
            # metric_dict[f'val/{data_source}/tool_call_count/min'] = np.min(tool_calls)

        for k, v in success_rate.items():
            metric_dict[f'val/{k}'] = v

        # === Skill Bank 动态更新 ===
        if self.config.env.get('skills_only_memory', {}).get('enable_dynamic_update', False):
            self._update_skills_from_validation(
                sample_inputs=sample_inputs,
                sample_outputs=sample_outputs,
                sample_scores=sample_scores,
                success_rate=success_rate,
            )

        return metric_dict

    def _update_skills_from_validation(
        self,
        sample_inputs: list,
        sample_outputs: list,
        sample_scores: list,
        success_rate: dict,
    ):
        """
        根据 validation 结果更新 skill bank。

        仅在特定任务类型成功率低于阈值时触发更新。
        """
        update_config = self.config.env.skills_only_memory
        threshold = update_config.get('update_threshold', 0.5)

        # 检查是否需要更新（某个任务类型成功率低于阈值）
        needs_update = False
        low_success_tasks = []
        for task_key, rate in success_rate.items():
            if rate < threshold:
                needs_update = True
                # 从 key 提取 task_type (e.g., "pick_and_place_success_rate" -> "pick_and_place")
                task_type = task_key.replace('_success_rate', '')
                low_success_tasks.append(task_type)

        if not needs_update:
            print(f"[SkillUpdate] All task success rates above {threshold}, skipping update")
            return

        print(f"[SkillUpdate] Low success tasks: {low_success_tasks}, triggering skill update...")

        # 收集失败 trajectories
        failed_trajectories = self._collect_failed_trajectories(
            sample_inputs, sample_outputs, sample_scores
        )

        if not failed_trajectories:
            print("[SkillUpdate] No failed trajectories found")
            return

        # 初始化 SkillUpdater (lazy init, 使用 Azure OpenAI o3)
        if not hasattr(self, 'skill_updater'):
            from agent_system.memory.skill_updater import SkillUpdater
            self.skill_updater = SkillUpdater(
                max_new_skills_per_update=update_config.get('max_new_skills', 3),
            )

        # 获取当前 skills
        retrieval_memory = self.val_envs.retrieval_memory
        if retrieval_memory is None:
            print("[SkillUpdate] No retrieval_memory found in val_envs")
            return

        # 分析失败并生成新 skills
        print(f"[SkillUpdate] Analyzing {len(failed_trajectories)} failed trajectories with o3...")
        new_skills = self.skill_updater.analyze_failures(
            failed_trajectories=failed_trajectories,
            current_skills=retrieval_memory.skills,
        )

        if new_skills:
            # Add to training envs only.
            # Do NOT add to val_envs here: skills derived from validation
            # failures must not be fed back into the validation memory of the
            # same evaluation cycle — that would create a data-leakage loop
            # where val scores are inflated by skills specifically targeting
            # the val set.
            if hasattr(self, 'envs') and hasattr(self.envs, 'retrieval_memory') and self.envs.retrieval_memory:
                self.envs.retrieval_memory.add_skills(new_skills, category='general')
                print(f"[SkillUpdate] Added {len(new_skills)} new skills to training envs")

            # Save updated skill bank (from training envs) to disk.
            train_memory = self.envs.retrieval_memory if (
                hasattr(self, 'envs') and hasattr(self.envs, 'retrieval_memory')
                and self.envs.retrieval_memory
            ) else retrieval_memory
            save_dir = self.config.trainer.get('default_local_dir', './outputs')
            save_path = os.path.join(save_dir, f'updated_skills_step{self.global_steps}.json')
            train_memory.save_skills(save_path)
            print(f"[SkillUpdate] Saved updated skill bank to {save_path}")
        else:
            print("[SkillUpdate] No new skills generated")

    def _collect_failed_trajectories(
        self,
        inputs: list,
        outputs: list,
        scores: list,
    ) -> list:
        """收集失败的 trajectories 用于分析"""
        failed = []
        for inp, out, score in zip(inputs, outputs, scores):
            if score <= 0:  # 失败的 trajectory
                task_type = self._detect_task_type_from_input(inp)
                task_desc = self._extract_task_description(inp)
                trajectory = self._parse_conversation_to_steps(inp, out)
                failed.append({
                    'task': task_desc,
                    'trajectory': trajectory,
                    'task_type': task_type,
                })
        return failed[:10]  # 限制数量，避免 prompt 过长

    def _extract_task_description(self, inp: str) -> str:
        """Extract the task description from a full conversation prompt."""
        import re
        # Common patterns used in ALFWorld, WebShop, OpenClaw, etc.
        patterns = [
            r'(?:Your task is to|Task:|task is to|you need to)[:\s]+(.*?)(?:\n|$)',
            r'(?:goal|objective)[:\s]+(.*?)(?:\n|$)',
        ]
        for pat in patterns:
            m = re.search(pat, inp, re.IGNORECASE)
            if m:
                return m.group(1).strip()[:1000]
        # Fallback: first user turn (skip system prompt)
        for marker in ('<|im_start|>user\n', '\nHuman: ', '\nUser: '):
            idx = inp.find(marker)
            if idx >= 0:
                start = idx + len(marker)
                return inp[start:start + 1000]
        return inp[:1000]

    def _parse_conversation_to_steps(self, inp: str, out: str) -> list:
        """
        Parse a full decoded conversation into a list of trajectory steps.

        Each step is ``{'action': str, 'observation': str}`` where
        ``observation`` is the environment feedback (user/tool turn) and
        ``action`` is the agent response (assistant turn).

        Falls back to treating the whole ``inp`` as the initial context when
        no structured turn markers are found.
        """
        import re
        steps = []

        # --- ChatML / Qwen format -------------------------------------------
        user_turns = re.findall(
            r'<\|im_start\|>user\n(.*?)<\|im_end\|>', inp, re.DOTALL
        )
        asst_turns = re.findall(
            r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>', inp, re.DOTALL
        )
        if user_turns and asst_turns:
            for obs, act in zip(user_turns, asst_turns):
                steps.append({
                    'action': act.strip()[:1500],
                    'observation': obs.strip()[:800],
                })
            # Final (failed) action has no follow-up observation
            steps.append({'action': out[:2000], 'observation': ''})
            return steps

        # --- Human / Assistant format ----------------------------------------
        user_turns = re.findall(
            r'(?:Human|User):\s*(.*?)(?=(?:Human|User|Assistant):|$)',
            inp, re.DOTALL | re.IGNORECASE,
        )
        asst_turns = re.findall(
            r'Assistant:\s*(.*?)(?=(?:Human|User|Assistant):|$)',
            inp, re.DOTALL | re.IGNORECASE,
        )
        if user_turns and asst_turns:
            for obs, act in zip(user_turns, asst_turns):
                steps.append({
                    'action': act.strip()[:1500],
                    'observation': obs.strip()[:800],
                })
            steps.append({'action': out[:2000], 'observation': ''})
            return steps

        # --- Fallback: treat full inp as initial context ---------------------
        steps.append({'action': '', 'observation': inp[:3000]})
        steps.append({'action': out[:2000], 'observation': ''})
        return steps

    def _detect_task_type_from_input(self, inp: str) -> str:
        """Deprecated: ALFWorld task_type should come from gamefile, not prompt text.
        Returns 'pick_and_place' as fallback."""
        print("[SkillEvolution] WARNING: _detect_task_type_from_input called; "
              "task_type should be derived from gamefile, not prompt keywords.")
        return 'pick_and_place'

    # ------------------------------------------------------------------ #
    # Skill evolution (A/B rollout: meta-dim update + prune)              #
    # ------------------------------------------------------------------ #

    def _init_skill_evolution(self):
        """Lazily initialise skill memory, updater, and pruner if configured."""
        skill_cfg = self.config.get('skill', {})
        if not skill_cfg.get('ab_rollout', False):
            self._skill_memory = None
            return

        from agent_system.memory.skills_only_memory import SkillsOnlyMemory
        from agent_system.memory.skill_updater import SkillUpdater
        from agent_system.memory.skill_pruner import SkillPruner

        skills_path = skill_cfg.get(
            'skills_json_path',
            self.config.env.get('skills_only_memory', {}).get(
                'skills_json_path', 'memory_data/alfworld/claude_style_skills.json'
            ),
        )
        retrieval_mode = self.config.env.get('skills_only_memory', {}).get('retrieval_mode', 'template')
        top_k = self.config.env.get('skills_only_memory', {}).get('top_k', 6)

        self._skill_memory = SkillsOnlyMemory(
            skills_json_path=skills_path,
            retrieval_mode=retrieval_mode,
        )
        self._skill_top_k = top_k
        self._skill_llm_model = skill_cfg.get('llm_model') or os.getenv('SKILLRL_LLM_MODEL', 'claude-sonnet-4-6')
        self._skill_updater = SkillUpdater(
            max_new_skills_per_update=skill_cfg.get('max_new_skills', 3),
            model=self._skill_llm_model,
        )
        self._skill_pruner = SkillPruner(
            min_utility=skill_cfg.get('min_utility', 0.2),
            merge_threshold=skill_cfg.get('merge_threshold', 0.85),
        )
        self._skill_save_path = skill_cfg.get('save_path', skills_path)
        self._skill_update_freq = skill_cfg.get('update_freq', 10)
        self._skill_update_lower_bound = skill_cfg.get('update_lower_bound', 0.0)
        self._skill_prune_threshold = skill_cfg.get('prune_threshold', 0.6)
        # Number of val episodes used to verify skill changes (0 = skip verification)
        self._skill_val_episodes = skill_cfg.get('val_episodes', 16)
        # Verification method: 'rollout' (true causal replay) or 'notverify' (skip verification)
        self._skill_verify_method = skill_cfg.get('verify_method', 'rollout')
        # Dimension selection mode: 'fixed' (SR-based preset) or 'auto' (LLM free choice)
        self._skill_focus_dims_mode = skill_cfg.get('focus_dims_mode', 'fixed')
        self._last_skill_sr = None

        # Anchor buffer removed: anchor verify is statistically unsound
        # (random step sampling, _score_actions fallback=0, no task alignment).
        # Only rollout verify is supported.
        sc = self._skill_memory.get_skill_count()
        print(
            f"[SkillEvol init] skill_memory loaded: {sc['total']} skills "
            f"(general={sc.get('general',0)}, dynamic={sc.get('dynamic',0)}) "
            f"from {skills_path}"
        )
        print(
            f"[SkillEvol init] skill_updater: model={self._skill_llm_model}, "
            f"max_new_skills={skill_cfg.get('max_new_skills', 3)}"
        )
        print(
            f"[SkillEvol init] skill_pruner: min_utility={skill_cfg.get('min_utility', 0.2)}, "
            f"merge_threshold={skill_cfg.get('merge_threshold', 0.85)}, "
            f"prune_threshold={self._skill_prune_threshold}"
        )
        print(
            f"[SkillEvol init] schedule: update_freq={self._skill_update_freq}, "
            f"lower_bound={self._skill_update_lower_bound}, "
            f"val_episodes={self._skill_val_episodes}, "
            f"verify_method={self._skill_verify_method}, focus_dims_mode={self._skill_focus_dims_mode}"
        )
        # Verify LLM API connectivity from inside the Ray worker (rank0).
        # This is the correct place: shell-level tests fail on Hope cluster
        # because only Ray workers have outbound network access.
        self._test_llm_api()
        # Tag updater's client so synthesize_skill can resolve the model name.
        self._skill_updater.client._skill_model = self._skill_llm_model

    def _test_llm_api(self):
        """Verify LLM API connectivity from inside the Ray worker."""
        import os, sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
        from agent_system.memory.skill_updater import _llm_chat

        model  = getattr(self, '_skill_llm_model', 'claude-sonnet-4-6')
        mis_id = os.getenv("SKILLRL_MIS_ID") or os.getenv("CATPAW_MIS_ID", "YOUR_MIS_ID")
        try:
            content = _llm_chat(
                messages=[{"role": "user", "content": "Reply with the single word: OK"}],
                model=model, mis_id=mis_id, timeout=30,
            )
            if content:
                print(f"[SkillEvolution] LLM API OK — model={model} response='{content.strip()}'")
            else:
                print(f"[SkillEvolution] LLM API reachable but empty content — skill meta updates may be skipped")
        except Exception as e:
            print(f"[SkillEvolution] LLM API unreachable: {e} — skill meta updates will be skipped")

    def _run_skill_evolution(self, batch, step: int, gen_batch=None) -> dict:
        """Post-rollout skill update + prune.  Called every _skill_update_freq steps."""
        if getattr(self, '_skill_memory', None) is None:
            return {}

        skill_sr = batch.meta_info.get('skill_sr')
        noskill_sr = batch.meta_info.get('noskill_sr')
        if skill_sr is None:
            return {}
        if noskill_sr is None:
            noskill_sr = 0.0

        sc = self._skill_memory.get_skill_count()
        metrics = {
            'skill/skill_sr': float(skill_sr),
            'skill/noskill_sr': float(noskill_sr),
            'skill/skill_count': sc['total'],
        }

        print(f"\n[SkillEvol step={step}] ===== skill evolution triggered =====")
        print(f"[SkillEvol step={step}] input SR: skill_sr={skill_sr:.3f}, noskill_sr={noskill_sr:.3f}")
        print(f"[SkillEvol step={step}] skill_bank: {sc['total']} skills "
              f"(general={sc.get('general',0)}, dynamic={sc.get('dynamic',0)})")

        # Collect per-task-type SR from meta_info and surface to metrics
        per_type_skill_sr: dict = {}
        for key, val in batch.meta_info.items():
            if key.startswith('skill_sr_'):
                tt = key[len('skill_sr_'):]
                per_type_skill_sr[tt] = float(val)
                metrics[f'skill/skill_sr_{tt}'] = float(val)
            elif key.startswith('noskill_sr_'):
                tt = key[len('noskill_sr_'):]
                metrics[f'skill/noskill_sr_{tt}'] = float(val)
        if per_type_skill_sr:
            per_type_str = ', '.join(f'{tt}={v:.3f}' for tt, v in sorted(per_type_skill_sr.items()))
            print(f"[SkillEvol step={step}] SR by task_type: {per_type_str}")

        # Save snapshot BEFORE any changes; also record the pre-change skill_sr
        # as the baseline for next round's rollback check.
        snapshot = self._skill_memory.snapshot()
        changed = False
        # Track which skill IDs were changed/added per task_type for per-type rollback.
        # 'general' key holds IDs of general skills that were updated.
        changed_by_type: Dict[str, List[str]] = {}

        # Prune: no-skill arm doing well → skills may be redundant
        if noskill_sr > self._skill_prune_threshold:
            print(f"[SkillEvol step={step}] prune: noskill_sr={noskill_sr:.3f} > "
                  f"threshold={self._skill_prune_threshold} → running")
            nb = batch.non_tensor_batch
            # Collect replay states here so prune-only verify uses current-step data
            self._collect_replay_states_for_verify(nb)
            success_trajs = self._extract_trajs_for_skill(nb, with_skill=True, success_value=True)
            all_ids = self._extract_skill_ids_for_skill(nb, with_skill=True)
            attribution = self._skill_pruner.attribute_from_successes(success_trajs)
            print(f"[SkillEvol step={step}] prune input: {len(success_trajs)} success_trajs, "
                  f"{len(all_ids)} retrieved_ids")
            print(f"[SkillEvol step={step}] prune attribution: "
                  f"{ {sid: cnt for sid, cnt in sorted(attribution.items(), key=lambda x: -x[1])[:10]} }")
            evicted = self._skill_pruner.prune(
                self._skill_memory, attribution, all_skill_ids=all_ids
            )
            sc_after = self._skill_memory.get_skill_count()
            if evicted:
                metrics['skill/evicted'] = len(evicted)
                changed = True
                # Track evicted ids so verify uses them as changed_by_type['general']
                changed_by_type.setdefault('general', []).extend(
                    [sid for sid in evicted if sid not in changed_by_type.get('general', [])]
                )
                print(f"[SkillEvol step={step}] prune result: evicted={evicted}, "
                      f"bank={sc_after['total']} skills remaining")
            else:
                print(f"[SkillEvol step={step}] prune result: nothing evicted, "
                      f"bank={sc_after['total']} skills unchanged")
        else:
            print(f"[SkillEvol step={step}] prune: noskill_sr={noskill_sr:.3f} <= "
                  f"threshold={self._skill_prune_threshold} → skipped")

        # Anchor buffer removed: update_from_batch no longer called.

        # Update: skill_sr in (lower_bound, 1.0); focus dims are sr-adaptive
        if self._skill_update_lower_bound <= skill_sr < 1.0:
            print(f"[SkillEvol step={step}] update check: skill_sr={skill_sr:.3f} in "
                  f"[{self._skill_update_lower_bound}, 1.0) → proceed")
            nb = batch.non_tensor_batch
            failed_trajs = self._extract_trajs_for_skill(nb, with_skill=True, success_value=False)
            success_trajs_for_log = self._extract_trajs_for_skill(nb, with_skill=True, success_value=True)
            # Collect replay states before skill update for causal verification
            self._collect_replay_states_for_verify(nb)

            failed_by_type = {}
            for t in failed_trajs:
                tt = t.get('task_type') or 'unknown'
                failed_by_type[tt] = failed_by_type.get(tt, 0) + 1
            success_by_type = {}
            for t in success_trajs_for_log:
                tt = t.get('task_type') or 'unknown'
                success_by_type[tt] = success_by_type.get(tt, 0) + 1

            # Collect all skill_ids seen in failed trajs
            failed_skill_freq: dict = {}
            for t in failed_trajs:
                for sid in (t.get('skill_ids_used') or []):
                    failed_skill_freq[sid] = failed_skill_freq.get(sid, 0) + 1
            top_failed_skills = sorted(failed_skill_freq.items(), key=lambda x: -x[1])[:8]

            print(f"[SkillEvol step={step}] traj extract (skill_arm):")
            print(f"  success: {len(success_trajs_for_log)} trajs, by_type={success_by_type}")
            print(f"  failed:  {len(failed_trajs)} trajs, by_type={failed_by_type}")
            print(f"  skill_ids in failed (top-8 by freq): {top_failed_skills}")

            if failed_trajs:
                # Group failed/success trajectories by task_type
                groups: dict = {}
                for traj in failed_trajs:
                    tt = traj.get('task_type') or 'unknown'
                    groups.setdefault(tt, []).append(traj)

                # Also group success trajs by type for contrast
                success_by_type_for_llm: dict = {}
                for traj in success_trajs_for_log:
                    tt = traj.get('task_type') or 'unknown'
                    success_by_type_for_llm.setdefault(tt, []).append(traj)

                # Prefix map: task_type → skill_id prefix for new dynamic skills
                _TYPE_PREFIX = {
                    'pick_and_place': 'pic', 'pick_two': 'ptw',
                    'look_at_obj_in_light': 'loo', 'clean': 'cle',
                    'heat': 'hea', 'cool': 'coo', 'examine': 'exa',
                }

                all_updates: List[dict] = []
                total_added = 0

                for tt, group_trajs in groups.items():
                    tt_sr = per_type_skill_sr.get(tt, skill_sr)
                    # focus_dims_mode: 'fixed' uses SR-based preset; 'auto' lets LLM choose
                    if self._skill_focus_dims_mode == 'auto':
                        focus_dims = None
                    else:
                        focus_dims = self._get_focus_dims(tt_sr)
                    tt_success_trajs = success_by_type_for_llm.get(tt, [])[:2]
                    used_ids: set = set()
                    for traj in group_trajs:
                        used_ids.update(traj.get('skill_ids_used') or [])
                    print(f"[SkillEvol step={step}] llm_call: task_type='{tt}', "
                          f"n_failed={len(group_trajs)}, n_success_ref={len(tt_success_trajs)}, "
                          f"skill_sr={tt_sr:.3f}, focus_dims={focus_dims or 'auto'}, "
                          f"used_skill_ids={sorted(used_ids)}")
                    import time as _time
                    _t0 = _time.time()
                    result = self._skill_updater.analyze_failures_with_meta(
                        group_trajs, self._skill_memory.skills,
                        focus_dims=focus_dims,
                        success_trajs=tt_success_trajs,
                    )
                    _elapsed = _time.time() - _t0
                    # Rule: only apply updates for skills that were actually used
                    filtered_updates = [
                        u for u in result['updates'] if u['skill_id'] in used_ids
                    ]
                    skipped = len(result['updates']) - len(filtered_updates)
                    print(f"[SkillEvol step={step}] llm_resp: task_type='{tt}', elapsed={_elapsed:.1f}s")
                    print(f"  updates (raw={len(result['updates'])}): "
                          f"{[(u['skill_id'], u['dim']) for u in result['updates']]}")
                    if skipped:
                        dropped = [u['skill_id'] for u in result['updates'] if u['skill_id'] not in used_ids]
                        print(f"  updates dropped (not in used_ids, n={skipped}): {dropped}")
                    print(f"  updates (filtered={len(filtered_updates)}): "
                          f"{[(u['skill_id'], u['dim']) for u in filtered_updates]}")
                    # fixed 模式下打印维度偏离率（实际 dim vs focus_dims）
                    if focus_dims and filtered_updates:
                        aligned = [u for u in filtered_updates if u['dim'] in focus_dims]
                        deviated = [u for u in filtered_updates if u['dim'] not in focus_dims]
                        if deviated:
                            print(f"  [fixed mode] dim alignment: {len(aligned)}/{len(filtered_updates)} "
                                  f"in focus_dims={focus_dims}, deviated: "
                                  f"{[(u['skill_id'], u['dim']) for u in deviated]}")

                    # Rename new skill IDs from dyn_NNN to {prefix}_NNN
                    prefix = _TYPE_PREFIX.get(tt, 'dyn')
                    renamed_new_skills = []
                    for s in result['new_skills']:
                        old_id = s.get('skill_id', '')
                        if old_id.startswith('dyn_'):
                            suffix = old_id[4:]  # keep NNN part
                            new_id = f"{prefix}_{suffix}"
                            s = dict(s, skill_id=new_id)
                        renamed_new_skills.append(s)
                    print(f"  new_skills ({len(renamed_new_skills)}): "
                          f"{[(s.get('skill_id'), s.get('title')) for s in renamed_new_skills]}")
                    all_updates.extend(filtered_updates)

                    # Record updates to changed_by_type
                    for u in filtered_updates:
                        # Determine whether this skill belongs to a task_type or general
                        skill_obj = self._skill_memory._find_skill(u['skill_id'])
                        skill_cat = 'general'
                        if skill_obj:
                            for ts_tt, ts_skills in self._skill_memory.skills.get('task_specific_skills', {}).items():
                                if any(s['skill_id'] == u['skill_id'] for s in ts_skills):
                                    skill_cat = ts_tt
                                    break
                        changed_by_type.setdefault(skill_cat, [])
                        if u['skill_id'] not in changed_by_type[skill_cat]:
                            changed_by_type[skill_cat].append(u['skill_id'])

                    # Add new skills with correct category and track in changed_by_type
                    if renamed_new_skills:
                        added = self._skill_memory.add_skills(renamed_new_skills, category=tt)
                        total_added += added
                        new_ids = [s.get('skill_id') for s in renamed_new_skills]
                        changed_by_type.setdefault(tt, [])
                        for s in renamed_new_skills:
                            sid = s.get('skill_id')
                            if sid and sid not in changed_by_type[tt]:
                                changed_by_type[tt].append(sid)
                        print(f"[SkillEvol step={step}] add_skills: task_type='{tt}', "
                              f"added={added}, ids={new_ids}, "
                              f"titles={[s.get('title') for s in renamed_new_skills]}")

                if total_added > 0:
                    metrics['skill/added'] = total_added
                    changed = True

                # Apply all updates and synthesize once
                print(f"[SkillEvol step={step}] apply updates: {len(all_updates)} total")
                for upd in all_updates:
                    skill_obj = self._skill_memory._find_skill(upd['skill_id'])
                    old_content = skill_obj.get('meta', {}).get(upd['dim'], '(none)') if skill_obj else '(not found)'
                    self._skill_memory.update_meta_dim(
                        upd['skill_id'], upd['dim'], upd['new_content']
                    )
                    print(f"  {upd['skill_id']} {upd['dim']}:")
                    print(f"    OLD: {old_content[:120]}")
                    print(f"    NEW: {upd['new_content'][:120]}")
                if all_updates:
                    updated_ids = list({u['skill_id'] for u in all_updates})
                    # also synthesize newly added skills (they have meta but no synthesized_text yet)
                    new_ids = [sid for ids in changed_by_type.values() for sid in ids
                               if sid not in {u['skill_id'] for u in all_updates}]
                    synthesize_ids = updated_ids + new_ids
                    self._skill_memory.synthesize_all(llm_client=self._skill_updater,
                                                      skill_ids=synthesize_ids)
                    import torch as _torch; _torch.cuda.empty_cache()
                    changed = True
                    print(f"[SkillEvol step={step}] synthesize_all: done for {len(synthesize_ids)} skills "
                          f"(updated={len(updated_ids)}, new={len(new_ids)})")
                sc_post = self._skill_memory.get_skill_count()
                print(f"[SkillEvol step={step}] skill_bank post-update: {sc_post['total']} skills "
                      f"(general={sc_post.get('general',0)}, dynamic={sc_post.get('dynamic',0)})")
                print(f"[SkillEvol step={step}] changed_by_type: {changed_by_type}")
                metrics['skill/meta_updates'] = len(all_updates)
        else:
            print(f"[SkillEvol step={step}] update check: skill_sr={skill_sr:.3f} not in "
                  f"({self._skill_update_lower_bound}, 1.0) → skipped")

        if changed:
            # --- Per-type skill verification and selective rollback ------------
            all_changed_ids = [sid for ids in changed_by_type.values() for sid in ids]
            if self._skill_verify_method == 'notverify':
                print(f"[SkillEvol step={step}] verify_method=notverify → skipping skill verification, accepting all changes")
                verify_result = None
            else:
                verify_result = self._verify_skill_change_replay(
                    metrics, all_changed_ids, rollout_batch=batch, gen_batch=gen_batch
                )
            post_sr_global, per_type_sr = verify_result if verify_result is not None else (None, {})
            pre_sr = metrics.get('skill/val_sr_before')

            if per_type_sr:
                # Per-type rollback: roll back only types where SR dropped > 0.05
                rolled_back_types = []
                accepted_types = []
                for tt, ids in changed_by_type.items():
                    if tt == 'general':
                        # General skills: use global SR as proxy
                        tt_pre  = pre_sr
                        tt_post = post_sr_global
                    else:
                        type_info = per_type_sr.get(tt)
                        if type_info is None:
                            # No verification data for this type → accept
                            accepted_types.append(tt)
                            continue
                        tt_pre  = type_info['pre']
                        tt_post = type_info['post']
                    if tt_pre is not None and tt_post is not None and tt_post < tt_pre - 0.05:
                        self._skill_memory.restore_partial(snapshot, ids)
                        rolled_back_types.append((tt, tt_pre, tt_post, ids))
                    else:
                        accepted_types.append(tt)

                metrics['skill/rollback'] = len(rolled_back_types)
                print(f"[SkillEvol step={step}] ===== done: PER-TYPE VERIFY =====")
                for tt, tp, tpp, ids in rolled_back_types:
                    print(f"  ROLLBACK {tt}: pre={tp:.3f} post={tpp:.3f} "
                          f"(Δ={tpp-tp:+.3f}) ids={ids}")
                for tt in accepted_types:
                    type_info = per_type_sr.get(tt, {})
                    tp  = type_info.get('pre',  pre_sr)
                    tpp = type_info.get('post', post_sr_global)
                    d_str = f"{tpp-tp:+.3f}" if tp is not None and tpp is not None else "N/A"
                    _tp_str  = f"{tp:.3f}"  if tp  is not None else "N/A"
                    _tpp_str = f"{tpp:.3f}" if tpp is not None else "N/A"
                    print(f"  ACCEPT  {tt}: pre={_tp_str} "
                          f"post={_tpp_str} (Δ={d_str}) "
                          f"ids={changed_by_type.get(tt, [])}")
            else:
                # No per-type data (anchor / val_envs fallback) → global decision
                rollback = (
                    post_sr_global is not None
                    and pre_sr is not None
                    and post_sr_global < pre_sr - 0.05
                )
                if rollback:
                    self._skill_memory.restore(snapshot)
                    metrics['skill/rollback'] = 1
                    print(f"[SkillEvol step={step}] ===== done: ROLLED BACK (global) =====")
                    print(f"  SR: {pre_sr:.3f} → {post_sr_global:.3f} "
                          f"(Δ={post_sr_global-pre_sr:+.3f} < -0.05)")
                else:
                    print(f"[SkillEvol step={step}] ===== done: ACCEPTED (global) =====")
                    _pre  = f"{pre_sr:.3f}" if pre_sr is not None else "N/A"
                    _post = f"{post_sr_global:.3f}" if post_sr_global is not None else "N/A"
                    print(f"  SR: {_pre} → {_post}")

            self._skill_memory.save_skills(self._skill_save_path)
            self._sync_skill_to_env(self.envs)
            sc_fin = self._skill_memory.get_skill_count()
            print(f"  skill_bank final: {sc_fin['total']} skills "
                  f"(general={sc_fin.get('general',0)}, "
                  f"task_specific={sc_fin.get('task_specific',0)})")
            print(f"  saved → {self._skill_save_path}")

        import torch as _torch
        _torch.cuda.empty_cache()
        return metrics

    @staticmethod
    def _get_focus_dims(skill_sr: float) -> List[str]:
        """Map global skill success rate to the meta dimensions LLM should focus on.

        Early training (SR < 0.5):  planning and exploration are the bottleneck → D-PLAN.
        Mid training (0.5–0.75):    state tracking and execution details → D-TRACK, D-EXEC.
        Late training (SR >= 0.75): fine-grained error prevention → D-GUARD.
        """
        if skill_sr < 0.5:
            dims = ['D-PLAN']
        elif skill_sr < 0.75:
            dims = ['D-TRACK', 'D-EXEC']
        else:
            dims = ['D-GUARD']
        print(f"[SkillEvolution] skill_sr={skill_sr:.3f} → focus_dims={dims}")
        return dims

    def _sync_skill_to_env(self, env_manager) -> None:
        """Copy current skill bank from _skill_memory into an env manager's retrieval_memory."""
        import copy as _copy
        env_mem = getattr(env_manager, 'retrieval_memory', None)
        if env_mem is not None:
            env_mem.skills = _copy.deepcopy(self._skill_memory.skills)
            env_mem._skill_embeddings_cache = None

    @staticmethod
    def _get_task_types_for_trajs(rollout_batch, traj_indices: List[int]) -> List[str]:
        """Return the distinct task_types for the given traj_index values."""
        nb = rollout_batch.non_tensor_batch
        traj_idx_arr  = nb.get('traj_index')
        task_type_arr = nb.get('task_type')
        if traj_idx_arr is None or task_type_arr is None:
            return []
        target = set(traj_indices)
        seen: set = set()
        types: List[str] = []
        for i in range(len(traj_idx_arr)):
            tid = int(traj_idx_arr[i])
            if tid not in target or tid in seen:
                continue
            seen.add(tid)
            tt = str(task_type_arr[i])
            if tt and tt not in types:
                types.append(tt)
        return types

    def _collect_replay_states_for_verify(self, nb) -> None:
        """Collect env state info from skill-arm trajectories for replay verification.

        Supports ALFWorld (gamefile), WebShop (session_idx), and Search (task_kwargs).

        pre_sr per state is the mean success over ALL rows with the same traj_index
        (i.e. all rollout_n skill-arm rollouts of that task), not just the first row.
        This reduces single-sample noise in the pre/post comparison.
        """
        max_states = max(1, getattr(self, '_skill_val_episodes', 16))
        wsm_arr         = nb.get('with_skills_mask')
        gamefile_arr    = nb.get('traj_gamefile')
        act_hist_arr    = nb.get('traj_action_history')
        session_idx_arr = nb.get('traj_session_idx')
        task_kwargs_arr = nb.get('traj_task_kwargs')
        traj_idx_arr    = nb.get('traj_index')
        task_type_arr   = nb.get('task_type')
        success_arr     = nb.get('traj_success')
        if wsm_arr is None:
            self._replay_states_for_verify = []
            return
        has_any_state = any(x is not None for x in [gamefile_arr, session_idx_arr, task_kwargs_arr])
        if not has_any_state:
            self._replay_states_for_verify = []
            return

        # Pass 1: collect one success value per unique traj_index (skill arm only)
        # traj_index is token-row level — deduplicate so each trajectory counts once.
        traj_successes: dict = {}  # tid -> list[bool]
        seen_tid_p1: set = set()
        for i in range(len(wsm_arr)):
            if not bool(wsm_arr[i]):
                continue
            if success_arr is None:
                continue
            tid = int(traj_idx_arr[i]) if traj_idx_arr is not None else i
            if tid in seen_tid_p1:
                continue
            seen_tid_p1.add(tid)
            traj_successes.setdefault(tid, []).append(bool(success_arr[i]))

        # Pass 2: build one state per unique TASK (gamefile/session_idx/task_kwargs),
        # pre_sr = mean over all traj_index rows that share the same task identity.
        # De-duplicate by task identity (not traj_index) so the same gamefile isn't
        # counted multiple times when rollout_n > 1.
        rollout_n = int(self.config.env.rollout.get('n', 8))
        # group traj_successes by task slot: task_slot = traj_index // rollout_n
        task_successes: dict = {}  # task_slot -> list[bool]
        for tid, s_list in traj_successes.items():
            task_slot = tid // rollout_n
            task_successes.setdefault(task_slot, []).extend(s_list)

        states = []
        seen_task: set = set()   # de-duplicate by task identity (gamefile/si/tk key)
        for i in range(len(wsm_arr)):
            if not bool(wsm_arr[i]):
                continue
            tid = int(traj_idx_arr[i]) if traj_idx_arr is not None else i
            gf = str(gamefile_arr[i])    if gamefile_arr    is not None else None
            si = int(session_idx_arr[i]) if session_idx_arr is not None and session_idx_arr[i] is not None else None
            tk = dict(task_kwargs_arr[i]) if task_kwargs_arr is not None and task_kwargs_arr[i] is not None else None
            # Use task identity as dedup key (explicit priority: gamefile > session_idx > task_kwargs > tid)
            if gf is not None:
                task_key = gf
            elif si is not None:
                task_key = ('si', si)
            elif tk is not None:
                task_key = str(tk)
            else:
                task_key = tid
            if task_key in seen_task:
                continue
            seen_task.add(task_key)
            ah = list(act_hist_arr[i])   if act_hist_arr    is not None and act_hist_arr[i] is not None else []
            tt = str(task_type_arr[i])   if task_type_arr   is not None else ''
            # pre_sr: mean over all rollouts of this task
            task_slot = tid // rollout_n
            all_s = task_successes.get(task_slot)
            pre_sr = float(np.mean(all_s)) if all_s else None
            states.append({'gamefile': gf, 'session_idx': si, 'task_kwargs': tk,
                           'action_history': ah, 'task_type': tt,
                           'success': pre_sr, 'traj_index': tid,
                           'pre_n': len(all_s) if all_s else 0})
            if len(states) >= max_states:
                break
        self._replay_states_for_verify = states
        env_types = []
        if any(s.get('gamefile') for s in states):               env_types.append('ALFWorld')
        if any(s.get('session_idx') is not None for s in states): env_types.append('WebShop')
        if any(s.get('task_kwargs') for s in states):            env_types.append('Search')
        avg_pre_n = float(np.mean([s['pre_n'] for s in states])) if states else 0
        print(f"[SkillVerify] Collected {len(states)} replay states "
              f"({'+'.join(env_types) or 'unknown env'}), avg pre_rollouts={avg_pre_n:.1f}")

    def _run_replay_rollouts(self, replay_states: list, gen_batch) -> list:
        """Run one rollout per replay state by resetting training envs to the same gamefile.

        Uses self.envs (training env) because replay gamefiles come from the training set;
        val_envs only loads valid_seen games and cannot load train-set paths.
        Training env state is fully reset at the start of each rollout step, so
        temporarily resetting it here does not corrupt the next training step.
        """
        import copy as _copy
        results = []
        envs = self.envs

        def _extract_sr_first_slot(val_output):
            """Extract success of the first trajectory only (slot 0), ignoring padding."""
            out_nb = val_output.non_tensor_batch
            # Prefer traj_success (bool per trajectory) at index 0
            ts = out_nb.get('traj_success')
            if ts is not None:
                arr = np.asarray(ts, dtype=np.float32)
                return float(arr[0]) if len(arr) > 0 else None
            # Fallback: find any success key, take first element
            _synthetic = {"success_rate_skill", "success_rate_origin"}
            for k in out_nb:
                if 'success' in k.lower() and k not in _synthetic:
                    arr = np.asarray(out_nb[k], dtype=np.float32)
                    return float(arr[0]) if len(arr) > 0 else None
            return None

        # Build env_kwargs list for all valid states at once
        valid_states = []
        env_kw_list  = []
        for state in replay_states:
            gf = state.get('gamefile')
            si = state.get('session_idx')
            tk = state.get('task_kwargs')
            if gf:
                env_kw_list.append({'gamefile': gf, 'force_skill_arm': True})
            elif si is not None:
                env_kw_list.append({'session_idx': si, 'force_skill_arm': True})
            elif tk:
                kw = dict(tk); kw['force_skill_arm'] = True
                env_kw_list.append(kw)
            else:
                continue
            valid_states.append(state)

        if not valid_states:
            return results

        # envs.reset_with_gamefile only resets worker[0]; pad the gen_batch to
        # envs.num_processes so multi_turn_loop sees the right batch size.
        env_batch_size = getattr(getattr(envs, 'envs', None), 'num_processes', 1)
        env_batch_size = max(env_batch_size, 1)

        self._sync_skill_to_env(envs)
        for idx, (state, env_kw) in enumerate(zip(valid_states, env_kw_list)):
            task_type = state.get('task_type', '')
            gf_label = env_kw.get('gamefile', str(env_kw))[-60:]
            try:
                # Pad gen_batch to env_batch_size, all rows use the same gamefile
                single_gen = _copy.deepcopy(gen_batch[:1])
                single_gen = single_gen.repeat(env_batch_size, interleave=False)
                single_gen.non_tensor_batch['env_kwargs'] = np.array(
                    [env_kw] * env_batch_size, dtype=object)
                val_output = self.traj_collector.multi_turn_loop(
                    gen_batch=single_gen,
                    actor_rollout_wg=self.actor_rollout_wg,
                    envs=envs,
                    is_train=False,
                )
                # Find the trajectory whose gamefile matches the pinned one
                out_nb = val_output.non_tensor_batch
                ts = out_nb.get('traj_success')
                actual_gf = out_nb.get('traj_gamefile')
                target_gf = env_kw.get('gamefile', '')
                sr_val = None
                matched_gf = '?'
                if ts is not None and actual_gf is not None:
                    for i in range(len(actual_gf)):
                        if str(actual_gf[i]) == target_gf:
                            sr_val = float(ts[i])
                            matched_gf = str(actual_gf[i])[-50:]
                            break
                # fallback: use index 0 if no match found
                if sr_val is None and ts is not None and len(ts) > 0:
                    sr_val = float(ts[0])
                    matched_gf = str(actual_gf[0])[-50:] if actual_gf is not None else '?'
                actual_tt = out_nb.get('task_type')
                tt0 = str(actual_tt[0]) if actual_tt is not None and len(actual_tt) > 0 else '?'
                print(f"[SkillVerify/Replay] [{idx}] expected={task_type}|...{gf_label[-40:]} "
                      f"got={tt0}|...{matched_gf} success={sr_val}")
                if sr_val is not None:
                    results.append({'success': sr_val > 0, 'task_type': task_type})
            except Exception as e:
                print(f'[SkillVerify/Replay] Replay failed for [{idx}] ...{gf_label}: {e}')
                continue
        return results

    def _verify_skill_change_replay(
        self,
        metrics: dict,
        changed_skill_ids: List[str] = None,
        rollout_batch=None,
        gen_batch=None,
    ) -> float:
        """Skill verification via true causal rollout (only supported method).

        Resets env to original training tasks and re-runs with new skill.
        pre_sr from stored training rollout results; post_sr from fresh rollout.
        """
        print(f"[SkillVerify] verify_method=rollout, changed_ids={changed_skill_ids}")

        # rollout verification
        replay_states = getattr(self, '_replay_states_for_verify', None)
        has_gamefiles = (
            replay_states and
            any(s.get('gamefile') for s in replay_states) and
            hasattr(self.envs, 'envs') and
            hasattr(self.envs.envs, 'reset_all_with_gamefiles')
        )
        has_session_idxs = (
            replay_states and
            any(s.get('session_idx') is not None for s in replay_states) and
            hasattr(self.envs, 'envs') and
            hasattr(self.envs.envs, 'reset_all_with_session_idxs')
        )
        if has_gamefiles or has_session_idxs:
            env_type = 'ALFWorld' if has_gamefiles else 'WebShop'
            n_states = len(replay_states) if replay_states else 0
            print(f"[SkillVerify/Rollout] method=true_replay ({env_type}), "
                  f"n_tasks={n_states}")
            import time as _time
            _t0 = _time.time()
            replay_result = self._verify_skill_change_true_replay(
                metrics, changed_skill_ids, rollout_batch, gen_batch, replay_states)
            _elapsed = _time.time() - _t0
            if replay_result is not None:
                post_sr_global, per_type_sr = replay_result
                pre = metrics.get('skill/val_sr_before')
                delta = post_sr_global - pre if pre is not None else float('nan')
                decision = "accept" if delta >= -0.05 else "ROLLBACK"
                _pre_str = f"{pre:.3f}" if pre is not None else "N/A"
                print(f"[SkillVerify/Rollout] elapsed={_elapsed:.1f}s, "
                      f"pre_sr={_pre_str}, "
                      f"post_sr={post_sr_global:.3f}, delta={delta:+.3f} → {decision}")
                # 展开 per_type_sr dict 为独立标量，避免 tensorboard 报 NotImplementedError
                for _tt, _info in per_type_sr.items():
                    if isinstance(_info, dict):
                        if 'post' in _info:
                            metrics[f'skill/val_per_type_sr/{_tt}'] = float(_info['post'])
                        if 'pre' in _info:
                            metrics[f'skill/val_per_type_sr_pre/{_tt}'] = float(_info['pre'])
                    else:
                        metrics[f'skill/val_per_type_sr/{_tt}'] = float(_info)
                return post_sr_global, per_type_sr
            print("[SkillVerify/Rollout] true_replay returned None, falling back to val_envs")

        # Final fallback: val_envs (no per-type breakdown available)
        print("[SkillVerify] method=val_envs fallback")
        post_sr = self._verify_skill_change(metrics, changed_skill_ids, rollout_batch, gen_batch)
        return post_sr, {}

    def _verify_skill_change_true_replay(
        self,
        metrics: dict,
        changed_skill_ids: List[str] = None,
        rollout_batch=None,
        gen_batch=None,
        replay_states: list = None,
    ) -> float:
        """True causal replay: reset each worker to its training task.

        Supports both ALFWorld (gamefile) and WebShop (session_idx).
        Uses reset_all_with_gamefiles / reset_all_with_session_idxs in parallel.
        Each task gets group_n=8 workers → reliable SR estimate per task.
        pre_sr: from training rollout (already computed).
        post_sr: same tasks, new skill text, same model weights.
        """
        import copy as _copy
        if not replay_states:
            return None

        # Detect env type and filter valid states
        alfworld_states = [s for s in replay_states if s.get('gamefile')]
        webshop_states  = [s for s in replay_states if s.get('session_idx') is not None]

        if alfworld_states:
            valid = alfworld_states
            env_key = 'gamefile'
            env_label = 'ALFWorld'
        elif webshop_states:
            valid = webshop_states
            env_key = 'session_idx'
            env_label = 'WebShop'
        else:
            return None

        # pre_sr from stored success rates
        pre_successes = [s['success'] for s in valid if s.get('success') is not None]
        pre_sr = float(np.mean(pre_successes)) if pre_successes else None
        pre_sr_str = f"{pre_sr:.3f}" if pre_sr is not None else "N/A"
        print(f"[SkillVerify/TrueReplay] {env_label} {len(valid)} tasks, pre_sr={pre_sr_str}")

        try:
            # Build env_kwargs: one dict per task (env_manager expands by group_n)
            rollout_n = int(self.config.env.rollout.get('n', 8))
            if env_key == 'gamefile':
                env_kwargs_list = [
                    {'gamefile': s['gamefile'], 'force_skill_arm': True}
                    for s in valid
                ]
            else:  # WebShop
                env_kwargs_list = [
                    {'session_idx': int(s['session_idx']), 'force_skill_arm': True}
                    for s in valid
                ]
            # gen_batch: repeat to match num_processes (n_tasks × rollout_n)
            # interleave=True: [task0]*rollout_n, [task1]*rollout_n, ...
            # so traj_index i → task_slot = i // rollout_n (correct)
            n_tasks = len(valid)
            replay_gen = _copy.deepcopy(gen_batch[:n_tasks])
            replay_gen = replay_gen.repeat(rollout_n, interleave=True)
            replay_gen.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }
            # env_kwargs: one entry per task (n_tasks), NOT per row.
            # env_manager.reset() reads kwargs as a list of task-level dicts.
            # The manager's reset() detects multiple entries and calls
            # reset_all_with_gamefiles / reset_all_with_session_idxs.
            replay_gen.non_tensor_batch['env_kwargs'] = np.array(
                env_kwargs_list, dtype=object)  # shape: (n_tasks,)

            self._sync_skill_to_env(self.envs)
            self.envs.val_rollout_always_skills = True
            val_output = self.traj_collector.multi_turn_loop(
                gen_batch=replay_gen,
                actor_rollout_wg=self.actor_rollout_wg,
                envs=self.envs,
                is_train=False,
            )
        except Exception as e:
            print(f"[SkillVerify/TrueReplay] Failed: {e}")
            self.envs.val_rollout_always_skills = False
            return None
        finally:
            self.envs.val_rollout_always_skills = False
            import torch as _torch
            _torch.cuda.empty_cache()

        # Extract post_sr: per-task mean over group_n rollouts
        out_nb = val_output.non_tensor_batch
        ts = out_nb.get('traj_success')
        traj_idx = out_nb.get('traj_index')
        # Use env-specific key for alignment check
        id_out = out_nb.get('traj_gamefile') if env_key == 'gamefile' \
                 else out_nb.get('traj_session_idx')
        if ts is None:
            return None

        ts_arr = np.asarray(ts, dtype=np.float32)
        # Group by task slot (traj_index // rollout_n), one entry per unique traj_index
        task_success: dict = {}
        seen_tid: set = set()
        for i in range(len(ts_arr)):
            tid = int(traj_idx[i]) if traj_idx is not None else i
            if tid in seen_tid:
                continue
            seen_tid.add(tid)
            task_slot = tid // rollout_n
            s, c = task_success.get(task_slot, (0, 0))
            task_success[task_slot] = (s + int(ts_arr[i] > 0), c + 1)

        # Print task-level results and build per-type SR dict
        print(f"[SkillVerify/TrueReplay] Task-level results ({env_label}):")
        # per_type_sr: {task_type: {'pre_sum': float, 'post_sum': float, 'n': int}}
        per_type_acc: dict = {}
        post_task_srs = []
        for i, state in enumerate(valid):
            s, c = task_success.get(i, (0, 0))
            task_sr = s / c if c > 0 else 0.0
            post_task_srs.append(task_sr)
            tt = state.get('task_type', 'unknown')
            pre_s_val = state.get('success', 0.0) or 0.0
            acc = per_type_acc.setdefault(tt, {'pre_sum': 0.0, 'post_sum': 0.0, 'n': 0})
            acc['pre_sum'] += pre_s_val
            acc['post_sum'] += task_sr
            acc['n'] += 1
            pre_s = f"{pre_s_val:.2f}"
            print(f"  [{tt}] pre={pre_s} "
                  f"post={task_sr:.2f} ({s}/{c}) {env_key}=...{str(state.get(env_key, '?'))[-40:]}")

        # Build per-type SR summary
        per_type_sr: Dict[str, Dict[str, float]] = {}
        for tt, acc in per_type_acc.items():
            n = acc['n']
            per_type_sr[tt] = {
                'pre':  acc['pre_sum']  / n,
                'post': acc['post_sum'] / n,
                'n':    n,
            }

        post_sr = float(np.mean(post_task_srs)) if post_task_srs else 0.0
        delta = post_sr - pre_sr if pre_sr is not None else float('nan')
        delta_str = f"{delta:+.3f}" if pre_sr is not None else "N/A"
        print(f"[SkillVerify/TrueReplay] pre_sr={pre_sr_str} "
              f"post_sr={post_sr:.3f} delta={delta_str}")
        for tt, v in sorted(per_type_sr.items()):
            d = v['post'] - v['pre']
            print(f"  {tt}: pre={v['pre']:.3f} post={v['post']:.3f} delta={d:+.3f} (n={int(v['n'])})")

        metrics['skill/val_sr_before'] = float(pre_sr) if pre_sr is not None else float('nan')
        metrics['skill/val_sr_after']  = post_sr
        metrics['skill/val_n_tasks']   = len(valid)
        metrics['skill/val_method']    = 3.0  # 3=true_replay
        return post_sr, per_type_sr

    def _rebuild_prompt_with_skills(self, obs: str, task_type: str, skill_memory) -> str:
        """Rebuild a prompt string with current skill text for anchor buffer verification.

        Retrieves skills from skill_memory for the given task_type, then formats
        them into the standard skill prompt template.
        """
        try:
            mem_cfg = self.config.env.get('skills_only_memory', {})
            top_k = mem_cfg.get('top_k', 6)
            task_top_k = mem_cfg.get('task_specific_top_k', 5)
            memories = skill_memory.retrieve(
                task_description=obs,
                top_k=top_k,
                task_type=task_type,
                task_specific_top_k=task_top_k,
            )
            # Format skills into text block (same format as env_manager prompt)
            skill_lines = []
            for s in memories.get('general_skills', []):
                skill_lines.append(f"- **{s.get('title','')}**: {s.get('synthesized','')}")
            for s in memories.get('task_specific_skills', []):
                skill_lines.append(f"- **{s.get('title','')}**: {s.get('synthesized','')}")
            skill_text = "\n".join(skill_lines)
            return f"{obs}\n\n## Retrieved Skills\n{skill_text}"
        except Exception:
            return obs

    def _verify_skill_change(
        self,
        metrics: dict,
        changed_skill_ids: List[str] = None,
        rollout_batch=None,
        gen_batch=None,
    ) -> float:
        """Verification rollout using the same tasks as the current training step.

        Selection: only tasks whose skill-arm trajectories used a changed skill.
        Retrieval: normal retrieve() flow (no force_include_ids), all trajectories
                   use the skill arm via val_rollout_always_skills=True.
        Rollout: 4 times per task (n_skill = rollout_n // 2).
        Rollback: compare post-change SR against pre-change SR of the same tasks.

        Returns None if verification is skipped.
        """
        if self._skill_val_episodes <= 0:
            return None
        if gen_batch is None or rollout_batch is None:
            return None

        rollout_n = int(self.config.env.rollout.get('n', 8))
        n_skill = max(1, rollout_n // 2)

        # ----------------------------------------------------------------
        # 1. Find tasks that used a changed skill (skill arm only)
        # ----------------------------------------------------------------
        nb = rollout_batch.non_tensor_batch
        traj_idx_arr  = nb.get('traj_index')
        wsm_arr       = nb.get('with_skills_mask')
        skill_ids_arr = nb.get('skill_ids_used')

        if traj_idx_arr is None or wsm_arr is None or skill_ids_arr is None:
            print("[SkillVerify] Missing traj metadata in batch, skipping verification")
            return None

        changed_set = set(changed_skill_ids or [])
        # traj_index is in range [0, batch_size*rollout_n).
        # gen_batch has batch_size rows (before repeat), so convert:
        #   task_idx = traj_index // rollout_n
        seen_traj: set = set()
        matched_task_set: set = set()   # task-level indices into gen_batch (0..n_tasks-1)
        matched_traj_indices: List[int] = []  # traj-level indices kept for _get_task_types_for_trajs
        for i in range(len(wsm_arr)):
            if not bool(wsm_arr[i]):
                continue
            tid = int(traj_idx_arr[i])
            if tid in seen_traj:
                continue
            seen_traj.add(tid)
            used = set(skill_ids_arr[i]) if skill_ids_arr[i] is not None else set()
            if used & changed_set:
                matched_traj_indices.append(tid)
                task_idx = tid // rollout_n
                matched_task_set.add(task_idx)

        matched_gen_indices: List[int] = sorted(matched_task_set)

        if not matched_traj_indices:
            print(f"[SkillVerify] No tasks used changed skills {list(changed_set)}, "
                  f"skipping verification")
            return None

        # ----------------------------------------------------------------
        # 2. Record pre-change SR for the matched tasks (from meta_info).
        #    Use per-task-type SR so the baseline is averaged over all rollouts
        #    of that task type (4 skill-arm rollouts), not a single trajectory.
        # ----------------------------------------------------------------
        matched_task_types = self._get_task_types_for_trajs(rollout_batch, matched_traj_indices)
        type_srs = [
            rollout_batch.meta_info[f'skill_sr_{tt}']
            for tt in matched_task_types
            if f'skill_sr_{tt}' in rollout_batch.meta_info
        ]
        pre_sr = float(np.mean(type_srs)) if type_srs else None
        # Fallback: use global skill_sr if per-type SR not available yet
        if pre_sr is None:
            pre_sr = rollout_batch.meta_info.get('skill_sr')
            if pre_sr is not None:
                pre_sr = float(pre_sr)
                print(f"[SkillVerify] per-type SR not found, using global skill_sr={pre_sr:.3f} as pre_sr")
        pre_sr_str = f"{pre_sr:.3f}" if pre_sr is not None else "N/A"
        print(f"[SkillVerify] {len(matched_gen_indices)} matched tasks (gen_indices={matched_gen_indices}) "
              f"task_types={matched_task_types} pre_sr={pre_sr_str}")

        # ----------------------------------------------------------------
        # 3. Build verification gen_batch.
        #
        #    val_envs has val_batch_size=128 independent environments (group_n=1),
        #    each reset() yields a different task.  We manually repeat gen_batch
        #    (16 tasks × rollout_n=8 = 128 rows) and pass is_train=False so
        #    multi_turn_loop does NOT repeat again.
        #    val_rollout_always_skills=True → all 128 rollouts use skills (no A/B).
        #    post_sr = mean over all 128 independent tasks.
        # ----------------------------------------------------------------
        if not hasattr(self, 'val_envs') or self.val_envs is None:
            print("[SkillVerify] val_envs not available, skipping verification")
            return None

        n_tasks = len(matched_gen_indices)
        # Manually repeat so val_envs receives exactly val_batch_size rows
        val_env_size = self.config.data.val_batch_size  # 128
        val_gen_batch = gen_batch.repeat(rollout_n, interleave=True)  # 16×8 = 128
        val_gen_batch.meta_info = {
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "recompute_log_prob": False,
            "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
            "validate": True,
        }

        # ----------------------------------------------------------------
        # 4. Sync updated skills to val_envs; all trajectories use skill arm.
        #    is_train=False → multi_turn_loop will NOT repeat() again.
        # ----------------------------------------------------------------
        self._sync_skill_to_env(self.val_envs)
        self.val_envs.val_rollout_always_skills = True
        print(f"[SkillVerify] Starting: {val_env_size} independent tasks via val_envs "
              f"(all skill arm), changed_skills={list(changed_set)}")

        try:
            val_output = self.traj_collector.multi_turn_loop(
                gen_batch=val_gen_batch,
                actor_rollout_wg=self.actor_rollout_wg,
                envs=self.val_envs,
                is_train=False,
            )
            if hasattr(self.val_envs, 'tasks') and self.val_envs.tasks:
                print(f"[SkillVerify] val_envs tasks ({len(self.val_envs.tasks)}):")
                for i, t in enumerate(self.val_envs.tasks[:val_env_size]):
                    print(f"  env={i}: {t}")
        except Exception as e:
            print(f"[SkillVerify] Verification rollout failed: {e}")
            return None
        finally:
            self.val_envs.val_rollout_always_skills = False
            import torch as _torch
            _torch.cuda.empty_cache()

        # ----------------------------------------------------------------
        # 5. Extract post-change SR over all 128 independent tasks.
        #    Deduplicate row-level → traj-level first.
        #    traj_index 0..127 maps back to gen_batch slot via gi = tid // rollout_n.
        # ----------------------------------------------------------------
        out_nb = val_output.non_tensor_batch
        _synthetic = {"success_rate_skill", "success_rate_origin"}
        sr_key = next(
            (k for k in out_nb if 'success' in k.lower() and k not in _synthetic
             and 'rate' in k.lower()),
            None,
        ) or next(
            (k for k in out_nb if 'success' in k.lower() and k not in _synthetic),
            None,
        )
        if sr_key is None:
            print("[SkillVerify] No success_rate key found in verification output")
            return None

        sr_raw = np.asarray(out_nb[sr_key], dtype=np.float32)
        traj_idx_out = out_nb.get('traj_index')
        task_type_out = out_nb.get('task_type')

        # Deduplicate: one entry per unique traj_index
        if traj_idx_out is not None:
            traj_idx_np = np.asarray(traj_idx_out, dtype=np.int64)
            seen_val: set = set()
            val_traj_sr: List[float] = []
            val_traj_tid: List[int] = []
            val_traj_tt: List[str] = []
            for row_i in range(len(traj_idx_np)):
                tid = int(traj_idx_np[row_i])
                if tid in seen_val:
                    continue
                seen_val.add(tid)
                val_traj_sr.append(float(sr_raw[row_i] > 0))
                val_traj_tid.append(tid)
                val_traj_tt.append(str(task_type_out[row_i]) if task_type_out is not None else '')
        else:
            val_traj_sr  = [float(v > 0) for v in sr_raw]
            val_traj_tid = list(range(len(val_traj_sr)))
            val_traj_tt  = [''] * len(val_traj_sr)

        # post_sr = mean success rate over all val_envs tasks (128 independent tasks).
        # Note: val_envs uses group_n=1 so traj_index mapping does NOT correspond to
        # gen_batch slots. We simply take the global average — this is a valid comparison
        # baseline because pre_sr is also a global average (per-type skill_sr).
        post_sr = float(np.mean(val_traj_sr)) if val_traj_sr else 0.0

        metrics['skill/val_sr_before'] = float(pre_sr) if pre_sr is not None else float('nan')
        metrics['skill/val_sr_after']  = post_sr
        metrics['skill/val_n_tasks']   = n_tasks
        delta = post_sr - pre_sr if pre_sr is not None else float('nan')
        _pre_str   = f"{pre_sr:.3f}"  if pre_sr  is not None else "N/A"
        _delta_str = f"{delta:+.3f}" if pre_sr  is not None else "N/A"
        n_val_trajs = len(val_traj_sr)
        print(f"[SkillVerify] Overall: pre_sr={_pre_str} post_sr={post_sr:.3f} "
              f"delta={_delta_str} matched_tasks={n_tasks} val_trajs={n_val_trajs}")
        return post_sr

    def _infer_batch_task_types(self, val_batch) -> List[str]:
        """Infer task type for each sample in val_batch.

        Uses ``data_source`` field if present (e.g. ``pick_clean_then_place``),
        otherwise falls back to decoding the raw prompt and running keyword detection.
        """
        n = len(val_batch)
        task_types = []

        data_sources = val_batch.non_tensor_batch.get('data_source')
        raw_prompts = val_batch.non_tensor_batch.get('raw_prompt')

        # data_source → task type mapping for ALFWorld
        # Categories must match skill bank keys: pick_and_place, look_at_obj_in_light,
        # clean, heat, cool, pick_two
        _ds_map = {
            'pick_clean_then_place_in_recep':    'clean',
            'pick_heat_then_place_in_recep':     'heat',
            'pick_cool_then_place_in_recep':     'cool',
            'look_at_obj_in_light':              'look_at_obj_in_light',
            'pick_two_obj_and_place':            'pick_two',
            'pick_and_place_simple':             'pick_and_place',
            'pick_and_place_with_movable_recep': 'pick_and_place',
        }

        for i in range(n):
            tt = None
            # Try data_source first (fastest)
            if data_sources is not None:
                ds = str(data_sources[i])
                for key, mapped in _ds_map.items():
                    if key in ds:
                        tt = mapped
                        break

            # data_source is the canonical source; if missing, fall back to
            # pick_and_place and warn (keyword detection removed for ALFWorld).
            if tt is None:
                print(f"[SkillEvolution] WARNING: could not determine task_type for "
                      f"sample {i} (no data_source); defaulting to 'pick_and_place'")
                tt = 'pick_and_place'

            task_types.append(tt)

        return task_types

    @staticmethod
    def _extract_trajs_for_skill(nb, with_skill: bool, success_value: bool) -> list:
        """Extract one trajectory dict per unique traj_index, filtered by arm and outcome.

        All arrays in non_tensor_batch are row-level (one entry per token row).
        We deduplicate by traj_index so each trajectory appears exactly once.
        """
        wsm_arr = nb.get('with_skills_mask')
        if wsm_arr is None:
            return []

        success_arr   = nb.get('traj_success')
        task_arr      = nb.get('task_description')
        task_type_arr = nb.get('task_type')
        skill_ids_arr = nb.get('skill_ids_used')
        traj_idx_arr  = nb.get('traj_index')
        traj_steps_arr = nb.get('trajectories')

        seen = set()
        trajs = []
        n = len(wsm_arr)
        for i in range(n):
            ws = bool(wsm_arr[i])
            sc = bool(success_arr[i]) if success_arr is not None else False
            if ws != with_skill or sc != success_value:
                continue
            # Deduplicate by traj_index; fall back to row index if absent
            tid = int(traj_idx_arr[i]) if traj_idx_arr is not None else i
            if tid in seen:
                continue
            seen.add(tid)
            steps = list(traj_steps_arr[i]) if (traj_steps_arr is not None
                                                  and traj_steps_arr[i] is not None) else []
            trajs.append({
                'task':           task_arr[i]       if task_arr       is not None else '',
                'task_type':      task_type_arr[i]  if task_type_arr  is not None else '',
                'trajectory':     steps,
                'skill_ids_used': list(skill_ids_arr[i]) if skill_ids_arr is not None and skill_ids_arr[i] is not None else [],
            })
        return trajs

    @staticmethod
    def _extract_skill_ids_for_skill(nb, with_skill: bool) -> list:
        wsm_arr = nb.get('with_skills_mask')
        skill_ids_arr = nb.get('skill_ids_used')
        if wsm_arr is None or skill_ids_arr is None:
            return []
        ids = []
        for i, ws in enumerate(wsm_arr):
            if bool(ws) == with_skill and skill_ids_arr[i] is not None:
                ids.extend(list(skill_ids_arr[i]))
        return ids

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy], config=self.config.actor_rollout_ref, role="ref")
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls, device_name=self.device_name, **wg_kwargs)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            self.async_rollout_mode = True
            self.async_rollout_manager = AsyncLLMServerManager(
                config=self.config.actor_rollout_ref,
                worker_group=self.actor_rollout_wg,
            )

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print("Warning: remove_previous_ckpt_in_save is deprecated," + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead")
        max_actor_ckpt_to_keep = self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        max_critic_ckpt_to_keep = self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1

        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep)

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt")
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst, k_partitions=world_size, equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # Initialise skill evolution (no-op if skill.ab_rollout not configured)
        self._init_skill_evolution()

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "env_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("env_kwargs")
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    # generate a batch
                    with _timer("gen", timing_raw):
                        # if not self.async_rollout_mode:
                        #     gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        # else:
                        #     self.async_rollout_manager.wake_up()
                        #     gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                        #     self.async_rollout_manager.sleep()

                        ################ agent-environment loop ###############
                        gen_batch_output = self.traj_collector.multi_turn_loop(
                                                                gen_batch=gen_batch,
                                                                actor_rollout_wg=self.actor_rollout_wg,
                                                                envs=self.envs,
                                                                is_train=True,
                                                                )
                    # Record GPU memory after rollout (gen) phase
                    import torch as _torch_mem
                    metrics["perf/mem_after_gen/allocated_gb"] = _torch_mem.cuda.max_memory_allocated() / (1024**3)
                    metrics["perf/mem_after_gen/reserved_gb"] = _torch_mem.cuda.max_memory_reserved() / (1024**3)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer("gen_max", timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    # batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                    # # repeat to align with repeated responses in rollout
                    # batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    # batch = batch.union(gen_batch_output)
                    del batch
                    batch = gen_batch_output

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.GiGPO:
                        step_rewards_tensor = core_gigpo.compute_step_discounted_returns(
                            batch=batch,
                            gamma=self.config.algorithm.gamma
                        )
                        batch.batch['step_rewards'] = step_rewards_tensor
                    
                    batch = adjust_batch(self.config, batch)

                    batch.batch["response_mask"] = compute_response_mask(batch)
                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with _timer("reward", timing_raw):
                        # compute reward model score
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # recompute old_log_probs
                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            rollout_old_log_probs = batch.batch["rollout_log_probs"]
                            actor_old_log_probs = batch.batch["old_log_probs"]
                            attention_mask = batch.batch["attention_mask"]
                            responses = batch.batch["responses"]
                            response_length = responses.size(1)
                            response_mask = attention_mask[:, -response_length:]

                            rollout_probs = torch.exp(rollout_old_log_probs)
                            actor_probs = torch.exp(actor_old_log_probs)
                            rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                            rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                            rollout_probs_diff_max = torch.max(rollout_probs_diff)
                            rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                            rollout_probs_diff_std = torch.std(rollout_probs_diff)
                            metrics.update(
                                {
                                    "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                                    "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                                    "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                                }
                            )

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer("ref", timing_raw):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        print(f"{list(reward_extra_infos_dict.keys())=}")
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_invalid_action_penalty if available
                        if self.config.actor_rollout_ref.actor.get('use_invalid_action_penalty', True):
                            batch, invalid_metrics = apply_invalid_action_penalty(batch,
                                                                                  invalid_action_penalty_coef=self.config.actor_rollout_ref.actor.invalid_action_penalty_coef,
                                                                                  )
                            metrics.update(invalid_metrics)

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # AHSR: Asymmetric Hindsight Skill Reward
                        _ahsr_cfg = getattr(self.config, 'skill', {}) if hasattr(self.config, 'skill') else {}
                        if isinstance(_ahsr_cfg, dict):
                            _ahsr_enabled  = _ahsr_cfg.get('ahsr_enabled', True)
                            _lambda_plus   = float(_ahsr_cfg.get('ahsr_lambda_plus',   2.0))
                            _lambda_minus  = float(_ahsr_cfg.get('ahsr_lambda_minus',  0.3))
                            _lambda_decay  = float(_ahsr_cfg.get('ahsr_lambda_decay',  4.0))
                            _lambda_min    = float(_ahsr_cfg.get('ahsr_lambda_min',    0.0))
                        else:
                            _ahsr_enabled  = getattr(_ahsr_cfg, 'ahsr_enabled', True)
                            _lambda_plus   = float(getattr(_ahsr_cfg, 'ahsr_lambda_plus',   2.0))
                            _lambda_minus  = float(getattr(_ahsr_cfg, 'ahsr_lambda_minus',  0.3))
                            _lambda_decay  = float(getattr(_ahsr_cfg, 'ahsr_lambda_decay',  4.0))
                            _lambda_min    = float(getattr(_ahsr_cfg, 'ahsr_lambda_min',    0.0))
                        if _ahsr_enabled:
                            batch, ahsr_metrics = apply_ahsr(batch, lambda_plus=_lambda_plus, lambda_minus=_lambda_minus, lambda_decay=_lambda_decay, lambda_min=_lambda_min)
                            metrics.update(ahsr_metrics)

                        # compute advantages, executed on the driver process

                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                            use_pf_ppo=self.config.algorithm.use_pf_ppo,
                            pf_ppo_reweight_method=self.config.algorithm.pf_ppo.reweight_method,
                            pf_ppo_weight_pow=self.config.algorithm.pf_ppo.weight_pow,
                            step_advantage_w=self.config.algorithm.gigpo.step_advantage_w,
                            gigpo_mode=self.config.algorithm.gigpo.mode,
                            gigpo_enable_similarity= self.config.algorithm.gigpo.enable_similarity,
                            gigpo_similarity_thresh=self.config.algorithm.gigpo.similarity_thresh,
                        )

                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer("update_actor", timing_raw):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with _timer("dump_rollout_generations", timing_raw):
                            print(batch.batch.keys())
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    # skill evolution hook (A/B rollout: update meta dims + prune)
                    if (
                        getattr(self, '_skill_memory', None) is not None
                        and self.global_steps % self._skill_update_freq == 0
                    ):
                        with _timer("skill_evolution", timing_raw):
                            skill_metrics = self._run_skill_evolution(
                                batch, self.global_steps, gen_batch=gen_batch
                            )
                            metrics.update(skill_metrics)
                        # Record GPU memory after skill_evolution phase
                        import torch as _torch_mem2
                        metrics["perf/mem_after_skill_evo/allocated_gb"] = _torch_mem2.cuda.max_memory_allocated() / (1024**3)
                        metrics["perf/mem_after_skill_evo/reserved_gb"] = _torch_mem2.cuda.max_memory_reserved() / (1024**3)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
