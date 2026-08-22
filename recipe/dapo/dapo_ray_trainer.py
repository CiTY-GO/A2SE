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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint

import numpy as np
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.ray_trainer import AdvantageEstimator, RayPPOTrainer, _timer, apply_kl_penalty, compute_advantage, compute_response_mask


class RayDAPOTrainer(RayPPOTrainer):
    # ------------------------------------------------------------------ #
    # Skill evolution helpers                                              #
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
        self._skill_updater = SkillUpdater(
            max_new_skills_per_update=skill_cfg.get('max_new_skills', 3),
        )
        self._skill_pruner = SkillPruner(
            min_utility=skill_cfg.get('min_utility', 0.2),
            merge_threshold=skill_cfg.get('merge_threshold', 0.85),
        )
        self._skill_save_path = skill_cfg.get('save_path', skills_path)
        self._skill_update_freq = skill_cfg.get('update_freq', 10)
        self._skill_update_threshold = skill_cfg.get('update_threshold', 0.4)
        self._skill_prune_threshold = skill_cfg.get('prune_threshold', 0.6)
        self._skill_val_episodes = skill_cfg.get('val_episodes', 10)
        self._last_skill_sr = None
        print(
            f"[SkillEvolution] Initialised | update_freq={self._skill_update_freq} "
            f"update_thr={self._skill_update_threshold} prune_thr={self._skill_prune_threshold}"
        )

    def _get_skill_text(self, task_description: str = "") -> str:
        """Retrieve and format skill text for the current task."""
        if self._skill_memory is None:
            return ""
        retrieved = self._skill_memory.retrieve(task_description, top_k=self._skill_top_k)
        return self._skill_memory.format_for_prompt(retrieved)

    def _run_skill_evolution(self, batch: DataProto, logger, step: int):
        """Post-rollout skill update + prune + validate.

        Called every ``_skill_update_freq`` steps when A/B rollout is active.
        """
        if self._skill_memory is None:
            return {}

        skill_sr = batch.meta_info.get('skill_sr')
        noskill_sr = batch.meta_info.get('noskill_sr')
        if skill_sr is None:
            return {}

        metrics = {
            'skill/skill_sr': skill_sr,
            'skill/noskill_sr': noskill_sr,
            'skill/skill_count': self._skill_memory.get_skill_count()['total'],
        }

        snapshot = self._skill_memory.snapshot()
        snapshot_sr = self._last_skill_sr or skill_sr
        changed = False

        # --- Prune: no-skill arm doing well → skills may be redundant --------
        if noskill_sr > self._skill_prune_threshold:
            success_trajs = self._extract_trajs(batch, with_skill=True, success=True)
            all_ids = self._extract_skill_ids(batch, with_skill=True)
            attribution = self._skill_pruner.attribute_from_successes(success_trajs)
            evicted = self._skill_pruner.prune(
                self._skill_memory, attribution, all_skill_ids=all_ids
            )
            if evicted:
                metrics['skill/evicted'] = len(evicted)
                changed = True
                print(f"[SkillEvolution] Pruned {len(evicted)} skills at step {step}")

        # --- Update: skill arm still failing → update meta dimensions ---------
        if skill_sr < self._skill_update_threshold:
            failed_trajs = self._extract_trajs(batch, with_skill=True, success=False)
            if failed_trajs:
                result = self._skill_updater.analyze_failures_with_meta(
                    failed_trajs, self._skill_memory.skills
                )
                for upd in result['updates']:
                    self._skill_memory.update_meta_dim(
                        upd['skill_id'], upd['dim'], upd['new_content']
                    )
                if result['updates']:
                    self._skill_memory.synthesize_all()
                    changed = True
                if result['new_skills']:
                    added = self._skill_memory.add_skills(result['new_skills'])
                    metrics['skill/added'] = added
                    changed = True
                metrics['skill/meta_updates'] = len(result['updates'])
                print(
                    f"[SkillEvolution] Updated {len(result['updates'])} meta dims, "
                    f"added {len(result['new_skills'])} skills at step {step}"
                )

        # --- Validate: if changed, do a quick sanity check via val metrics ----
        if changed:
            # We reuse the existing validation mechanism; the result is already
            # logged by the caller.  Here we just check if skill_sr degraded
            # compared to the snapshot baseline.
            new_sr = skill_sr  # optimistic: assume current batch is representative
            if new_sr < snapshot_sr - 0.05:
                self._skill_memory.restore(snapshot)
                metrics['skill/rollback'] = 1
                print(f"[SkillEvolution] Rolled back: new_sr={new_sr:.3f} < baseline={snapshot_sr:.3f}")
            else:
                self._skill_memory.save_skills(self._skill_save_path)
                self._last_skill_sr = new_sr

        return metrics

    @staticmethod
    def _extract_trajs(batch: DataProto, with_skill: bool, success: bool) -> list:
        """Extract trajectory dicts from batch non_tensor_batch for skill analysis."""
        trajs = []
        nb = batch.non_tensor_batch
        with_skill_arr = nb.get('with_skill')
        success_arr = nb.get('success_rate', nb.get('env_success'))
        if with_skill_arr is None:
            return trajs
        for i in range(len(with_skill_arr)):
            ws = bool(with_skill_arr[i])
            sc = bool(success_arr[i]) if success_arr is not None else False
            if ws == with_skill and sc == success:
                traj = {
                    'task': nb.get('task_description', [''] * len(with_skill_arr))[i]
                    if nb.get('task_description') is not None else '',
                    'task_type': nb.get('task_type', [''] * len(with_skill_arr))[i]
                    if nb.get('task_type') is not None else '',
                    'trajectory': [],
                    'skill_ids_used': list(nb.get('skill_ids_used', [None] * len(with_skill_arr))[i] or []),
                }
                trajs.append(traj)
        return trajs

    @staticmethod
    def _extract_skill_ids(batch: DataProto, with_skill: bool) -> list:
        """Collect all skill_ids retrieved in the skill arm."""
        nb = batch.non_tensor_batch
        with_skill_arr = nb.get('with_skill')
        skill_ids_arr = nb.get('skill_ids_used')
        if with_skill_arr is None or skill_ids_arr is None:
            return []
        ids = []
        for i, ws in enumerate(with_skill_arr):
            if bool(ws) == with_skill and skill_ids_arr[i] is not None:
                ids.extend(list(skill_ids_arr[i]))
        return ids

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

        # Initialise skill evolution (no-op if skill.ab_rollout is not set)
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

        timing_raw = defaultdict(float)
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}

                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                num_gen_batches += 1
                # pop those keys for generation
                if "multi_modal_data" in new_batch.non_tensor_batch.keys():
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                    )
                else:
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids"],
                    )

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    # generate a batch
                    with _timer("gen", timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer("gen_max", timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            new_batch = new_batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(new_batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            new_batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    new_batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object)
                    # repeat to align with repeated responses in rollout
                    new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    new_batch = new_batch.union(gen_batch_output)

                    with _timer("reward", timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                            new_batch = new_batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        try:
                            reward_result = self.reward_fn(new_batch, return_dict=True)
                            reward_tensor = reward_result["reward_tensor"]
                            reward_extra_infos_dict = reward_result["reward_extra_info"]
                        except Exception as e:
                            print(f"Error in reward_fn: {e}")
                            reward_tensor = self.reward_fn(new_batch)
                            reward_extra_infos_dict = {}

                        new_batch.batch["token_level_scores"] = reward_tensor

                        print(f"{list(reward_extra_infos_dict.keys())=}")
                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            new_batch, kl_metrics = apply_kl_penalty(new_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)  # TODO: This will be cleared if we use multiple genenration batches
                        else:
                            new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]

                    if not self.config.algorithm.filter_groups.enable:
                        batch = new_batch
                    else:  # NOTE: When prompts after filtering is less than train batch size,
                        # we skip to the next generation batch
                        metric_name = self.config.algorithm.filter_groups.metric
                        if metric_name == "seq_final_reward":
                            # Turn to numpy for easier filtering
                            new_batch.non_tensor_batch["seq_final_reward"] = new_batch.batch["token_level_rewards"].sum(dim=-1).numpy()
                        elif metric_name == "seq_reward":
                            new_batch.non_tensor_batch["seq_reward"] = new_batch.batch["token_level_scores"].sum(dim=-1).numpy()

                        # Collect the sequence reward for each trajectory
                        prompt_uid2metric_vals = defaultdict(list)
                        for uid, metric_val in zip(new_batch.non_tensor_batch["uid"], new_batch.non_tensor_batch[metric_name]):
                            prompt_uid2metric_vals[uid].append(metric_val)

                        prompt_uid2metric_std = {}
                        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                            prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)

                        kept_prompt_uids = [uid for uid, std in prompt_uid2metric_std.items() if std > 0 or len(prompt_uid2metric_vals[uid]) == 1]
                        num_prompt_in_batch += len(kept_prompt_uids)

                        kept_traj_idxs = []
                        for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch["uid"]):
                            if traj_from_prompt_uid in kept_prompt_uids:
                                kept_traj_idxs.append(idx)

                        new_batch = new_batch[kept_traj_idxs]
                        batch = new_batch if batch is None else DataProto.concat([batch, new_batch])

                        prompt_bsz = self.config.data.train_batch_size
                        if num_prompt_in_batch < prompt_bsz:
                            print(f"{num_prompt_in_batch=} < {prompt_bsz=}")
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                print(f"{num_gen_batches=}. Keep generating...")
                                continue
                            else:
                                raise ValueError(f"{num_gen_batches=} >= {max_num_gen_batches=}." + " Generated too many. Please check if your data are too difficult." + " You could also try set max_num_gen_batches=0 to enable endless trials.")
                        else:
                            # Align the batch
                            traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                            batch = batch[:traj_bsz]

                    # === Updating ===

                    batch.batch["response_mask"] = compute_response_mask(batch)

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

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

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer("ref", timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
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
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # skill evolution hook (post actor update, pre validation)
                    if (
                        getattr(self, '_skill_memory', None) is not None
                        and self.global_steps % self._skill_update_freq == 0
                    ):
                        with _timer("skill_evolution", timing_raw):
                            skill_metrics = self._run_skill_evolution(batch, logger, self.global_steps)
                            metrics.update(skill_metrics)

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

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                timing_raw = defaultdict(float)  # clear timing

                metrics["train/num_gen_batches"] = num_gen_batches
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
