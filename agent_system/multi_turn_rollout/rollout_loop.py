# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
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

import torch
import numpy as np
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from transformers import PreTrainedTokenizer
import uuid
from agent_system.multi_turn_rollout.utils import process_image, to_list_of_dict, torch_to_numpy, filter_group_data
from agent_system.environments import EnvironmentManagerBase
from typing import List, Dict
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

class TrajectoryCollector:
    def __init__(self, config, tokenizer: PreTrainedTokenizer, processor=None):
        """
        Initialize the TrajectoryProcessor class.
        
        Parameters:
            config: Configuration object containing data processing settings
            tokenizer (PreTrainedTokenizer): Tokenizer for text encoding and decoding
            processor: Image processor for multimodal inputs
        """
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor

    def preprocess_single_sample(
        self,
        item: int,
        gen_batch: DataProto,
        obs: Dict,
    ):
        """
        Process a single observation sample, organizing environment observations (text and/or images) 
        into a format processable by the model.
        
        Parameters:
            item (int): Sample index in the batch
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation, may contain 'text', 'image', 'anchor' keys
        
        Returns:
            dict: Contains processed input data such as input_ids, attention_mask, etc.
        """

        raw_prompt = gen_batch.non_tensor_batch['raw_prompt'][item]
        data_source = gen_batch.non_tensor_batch['data_source'][item]
        apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})
        
        # Get observation components
        obs_texts = obs.get('text', None)
        obs_images = obs.get('image', None)
        obs_anchors = obs.get('anchor', None)
        obs_text = obs_texts[item] if obs_texts is not None else None
        obs_image = obs_images[item] if obs_images is not None else None
        obs_anchor = obs_anchors[item] if obs_anchors is not None else None
        is_multi_modal = obs_image is not None

        _obs_anchor = torch_to_numpy(obs_anchor, is_object=True) if isinstance(obs_anchor, torch.Tensor) else obs_anchor

        # Build chat structure
        # obs_content = raw_prompt[0]['content']
        # if '<image>' in obs_content: 
        #     obs_content = obs_content.replace('<image>', '')

        # Build chat structure
        obs_content = ''
        if obs_text is not None:
            obs_content += obs_text
        else:
            print(f"Warning: No text observation found!")

        
        chat = np.array([{
            "content": obs_content,
            "role": "user",
        }])
        
        # Apply chat template
        prompt_with_chat_template = self.tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False,
            **apply_chat_template_kwargs
        )
        
        # Initialize return dict
        row_dict = {}
        
        # Process multimodal data
        if is_multi_modal:
            # Replace image placeholder with vision tokens
            raw_prompt = prompt_with_chat_template.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
            row_dict['multi_modal_data'] = {'image': [process_image(obs_image)]}
            image_inputs = self.processor.image_processor(row_dict['multi_modal_data']['image'], return_tensors='pt')
            image_grid_thw = image_inputs['image_grid_thw']
            row_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}
            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size**2
                index = 0
                while '<image>' in prompt_with_chat_template:
                    prompt_with_chat_template = prompt_with_chat_template.replace(
                        '<image>',
                        '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[index].prod() // merge_length) +
                        '<|vision_end|>',
                        1,
                    )
                    index += 1

                prompt_with_chat_template = prompt_with_chat_template.replace('<|placeholder|>',
                                                                                self.processor.image_token)

        else:
            raw_prompt = prompt_with_chat_template
        
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                            tokenizer=self.tokenizer,
                                                                            max_length=self.config.data.max_prompt_length,
                                                                            pad_token_id=self.tokenizer.pad_token_id,
                                                                            left_pad=True,
                                                                            truncation=self.config.data.truncation,)
        
        

        if is_multi_modal:

            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from verl.models.transformers.qwen3_vl import get_rope_index
            else:
                from verl.models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask[0],
            )  # (3, seq_length)
            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]  # (1, 4, seq_length)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.config.data.max_prompt_length:
            if self.config.data.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.config.data.max_prompt_length :]
            elif self.config.data.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.config.data.max_prompt_length]
            elif self.config.data.truncation == "middle":
                left_half = self.config.data.max_prompt_length // 2
                right_half = self.config.data.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.config.data.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.config.data.max_prompt_length}.")

        # Build final output dict
        row_dict.update({
            'input_ids': input_ids[0],
            'attention_mask': attention_mask[0],
            'position_ids': position_ids[0],
            'raw_prompt_ids': raw_prompt_ids,
            'anchor_obs': _obs_anchor,
            'index': item,
            'data_source': data_source
        })

        if self.config.data.get('return_raw_chat', False):
            row_dict['raw_prompt'] = chat.tolist()
        
        return row_dict

    def preprocess_batch(
        self,
        gen_batch: DataProto, 
        obs: Dict, 
    ) -> DataProto:
        """
        Process a batch of observation samples, converting environment observations into model-processable format.
        
        Parameters:
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation dictionary
                - 'text' (None or List[str]): Text observation data
                - 'image' (np.ndarray or torch.Tensor): Image observation data
                - 'anchor' (None or Any): Anchor observation without any histories or additional info. (for GiGPO only).
        
        Returns:
            DataProto: Contains processed batch data with preserved metadata
        """
        batch_size = len(gen_batch.batch['input_ids'])
        processed_samples = []
        
        # Process each sample in parallel
        for item in range(batch_size):
            # Extract per-sample observations
            processed = self.preprocess_single_sample(
                item=item,
                gen_batch=gen_batch,
                obs=obs,
            )
            processed_samples.append(processed)
        
        # Aggregate batch data
        batch = collate_fn(processed_samples)
        
        # Create DataProto with preserved metadata
        new_batch = DataProto.from_single_dict(
            data=batch,
            meta_info=gen_batch.meta_info
        )

        return new_batch


    def gather_rollout_data(
            self,
            total_batch_list: List[List[Dict]],
            episode_rewards: np.ndarray,
            episode_lengths: np.ndarray,
            success: Dict[str, np.ndarray],
            traj_uid: np.ndarray,
            tool_callings: np.ndarray,
            with_skills_per_traj: np.ndarray = None,
            ) -> DataProto:
        """
        Collect and organize trajectory data.

        Parameters:
            total_batch_list: List of trajectory data for each environment.
            episode_rewards:  Total rewards per environment.
            episode_lengths:  Total steps per environment.
            success:          Success metrics dict.
            traj_uid:         Trajectory unique identifiers.
            tool_callings:    Number of tool calls per environment.
            with_skills_per_traj: Boolean mask (batch_size,); True = skill arm.
                              None means all trajectories received skills.
        Returns:
            DataProto with collected trajectory data.
        """
        batch_size = len(total_batch_list)

        wsm_arr = None
        if with_skills_per_traj is not None:
            wsm_arr = np.asarray(with_skills_per_traj, dtype=bool).ravel()

        success_rate = {}
        for key, value in success.items():
            success_rate[key] = np.mean(value)

        # Find the primary per-trajectory success key from the environment
        # (excludes synthetic keys we add below to avoid circular reference).
        _synthetic = {"success_rate_skill", "success_rate_origin"}
        sr_key = next(
            (k for k in success if 'success' in k.lower() and k not in _synthetic),
            None,
        )
        per_traj_success = np.asarray(success[sr_key], dtype=np.float32).ravel() if sr_key else None

        # A/B split: compute per-arm success rates for skill evolution decisions
        if wsm_arr is not None and wsm_arr.shape[0] == batch_size and sr_key is not None:
            st = np.asarray(success[sr_key], dtype=np.float64).ravel()
            if st.shape[0] == batch_size:
                skill_vals = st[wsm_arr]
                origin_vals = st[~wsm_arr]
                success_rate["success_rate_skill"] = float(np.mean(skill_vals)) if skill_vals.size > 0 else float("nan")
                success_rate["success_rate_origin"] = float(np.mean(origin_vals)) if origin_vals.size > 0 else float("nan")

        effective_batch = []
        for bs in range(batch_size):
            for data in total_batch_list[bs]:
                assert traj_uid[bs] == data['traj_uid'], "data is not from the same trajectory"
                if data['active_masks']:
                    data['episode_rewards'] = episode_rewards[bs]
                    data['episode_lengths'] = episode_lengths[bs]
                    data['tool_callings'] = tool_callings[bs]
                    data['traj_index'] = bs
                    # Per-trajectory success bool for skill evolution filtering
                    if per_traj_success is not None and bs < len(per_traj_success):
                        data['traj_success'] = bool(per_traj_success[bs] > 0)
                    for key, value in success_rate.items():
                        data[key] = value
                    effective_batch.append(data)

        gen_batch_output = DataProto.from_single_dict(data=collate_fn(effective_batch))

        # Expand with_skills_mask to row level so adjust_batch / balance_batch never
        # see a length mismatch (mirrors D2Skill gather_rollout_data).
        if wsm_arr is not None:
            traj_idx = np.asarray(
                gen_batch_output.non_tensor_batch.get("traj_index"), dtype=np.int64
            ).ravel()
            gen_batch_output.non_tensor_batch["with_skills_mask"] = wsm_arr[traj_idx]

        return gen_batch_output

    def vanilla_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            ) -> DataProto:
        """
        Collects trajectories through parallel agent-environment agent_loop.
        Parameters:
            gen_batch (DataProto): Initial batch with prompts to start the agent_loop
            actor_rollout_wg (WorkerGroup): Worker group containing the actor model for policy decisions
            envs (EnvironmentManagerBase): Environment manager containing parallel environment instances
        
        Returns:
            total_batch_list (List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
        """

        batch_size = len(gen_batch.batch)

        # Initial observations from the environment
        obs, infos = envs.reset(kwargs=gen_batch.non_tensor_batch.pop('env_kwargs', None))

        lenght_obs = len(obs['text']) if obs['text'] is not None else len(obs['image'])
        assert len(gen_batch.batch) == lenght_obs, f"gen_batch size {len(gen_batch.batch)} does not match obs size {lenght_obs}"
        
        if self.config.env.rollout.n > 0: # env grouping
            uid_batch = []
            for i in range(batch_size):
                if i % self.config.env.rollout.n == 0:
                    uid = str(uuid.uuid4())
                uid_batch.append(uid)
            uid_batch = np.array(uid_batch, dtype=object)
        else: # no env grouping, set all to the same uid
            uid = str(uuid.uuid4())
            uid_batch = np.array([uid for _ in range(len(gen_batch.batch))], dtype=object)
        is_done = np.zeros(batch_size, dtype=bool)
        traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)
        total_batch_list = [[] for _ in range(batch_size)]
        total_infos = [[] for _ in range(batch_size)]
        # Per-trajectory action/observation steps for skill analysis
        traj_steps = [[] for _ in range(batch_size)]
        # Per-trajectory raw action history for replay-based verification
        traj_action_history = [[] for _ in range(batch_size)]
        # Per-trajectory admissible commands at each step (for anchor buffer)
        traj_admissible = [[] for _ in range(batch_size)]
        episode_lengths = np.zeros(batch_size, dtype=np.float32)
        episode_rewards = np.zeros(batch_size, dtype=np.float32)
        tool_callings = np.zeros(batch_size, dtype=np.float32)
        # Trajectory collection loop
        for _step in range(self.config.env.max_steps):
            active_masks = np.logical_not(is_done)

            batch = self.preprocess_batch(gen_batch=gen_batch, obs=obs)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            batch_input = batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            batch_input.meta_info = gen_batch.meta_info

            # pad to be divisible by dp_size
            batch_input_padded, pad_size = pad_dataproto_to_divisor(batch_input, actor_rollout_wg.world_size)
            batch_output_padded = actor_rollout_wg.generate_sequences(batch_input_padded)
            # # unpad
            batch_output = unpad_dataproto(batch_output_padded, pad_size=pad_size)

            batch.non_tensor_batch['uid'] = uid_batch
            batch.non_tensor_batch['traj_uid'] = traj_uid

            batch = batch.union(batch_output)
            
            text_actions = self.tokenizer.batch_decode(batch.batch['responses'], skip_special_tokens=True)

            next_obs, rewards, dones, infos = envs.step(text_actions)

            # Record (action, env_state) after step so observation is the
            # post-action environment feedback, not the full prompt.
            anchor_texts = next_obs.get('anchor') if next_obs is not None else None
            _infos_this_step = infos if infos is not None else [{}] * batch_size
            for i in range(batch_size):
                if not is_done[i]:
                    env_str = str(anchor_texts[i])[:300] if anchor_texts is not None else ''
                    info_i = _infos_this_step[i]
                    # Use projection result directly — env_manager sets projected_action after extraction
                    clean_action = info_i.get('projected_action', '')
                    valid = int(info_i.get('is_action_valid', 1 if clean_action else 0))

                    traj_steps[i].append({
                        'action':      clean_action,
                        'valid':       valid,
                        'observation': env_str,
                    })
            # Collect raw action history for replay-based verification
            for i in range(batch_size):
                if not is_done[i]:
                    traj_action_history[i].append(text_actions[i])

            # Collect admissible commands at this step for anchor buffer
            adm_cmds = None
            try:
                # get_admissible_commands is a @property returning prev_admissible_commands
                # Do NOT call it with (), just access it as a property
                if hasattr(envs, 'envs') and hasattr(envs.envs, 'get_admissible_commands'):
                    adm_cmds = envs.envs.get_admissible_commands  # @property, no ()
                elif hasattr(envs, 'get_admissible_commands'):
                    adm_cmds = envs.get_admissible_commands        # @property, no ()
            except Exception as _e:
                pass  # silently skip if not available
            for i in range(batch_size):
                if not is_done[i]:
                    cmds = list(adm_cmds[i]) if adm_cmds is not None and i < len(adm_cmds) else []
                    traj_admissible[i].append(cmds)

            
            if len(rewards.shape) == 2:
                rewards = rewards.squeeze(1)
            if len(dones.shape) == 2:
                # dones is numpy, delete a dimension
                dones = dones.squeeze(1)

            if 'is_action_valid' in infos[0]:
                batch.non_tensor_batch['is_action_valid'] = np.array([info['is_action_valid'] for info in infos], dtype=bool)
            else:
                batch.non_tensor_batch['is_action_valid'] = np.ones(batch_size, dtype=bool)

            if 'tool_calling' in infos[0]:
                tool_callings[active_masks] += np.array([info['tool_calling'] for info in infos], dtype=np.float32)[active_masks]
            # Create reward tensor, only assign rewards for active environments
            # episode_rewards += torch_to_numpy(rewards) * torch_to_numpy(active_masks)
            episode_rewards[active_masks] += torch_to_numpy(rewards)[active_masks]
            episode_lengths[active_masks] += 1

            assert len(rewards) == batch_size, f"env should return rewards for all environments, got {len(rewards)} rewards for {batch_size} environments"
            batch.non_tensor_batch['rewards'] = torch_to_numpy(rewards, is_object=True)
            batch.non_tensor_batch['active_masks'] = torch_to_numpy(active_masks, is_object=True)
            
            # Update episode lengths for active environments
            batch_list: list[dict] = to_list_of_dict(batch)

            for i in range(batch_size):
                total_batch_list[i].append(batch_list[i])
                total_infos[i].append(infos[i])

            # Update done states
            is_done = np.logical_or(is_done, dones)
                
            # Update observations for next step
            obs = next_obs

            # Break if all environments are done
            if is_done.all():
                break
        
        success: Dict[str, np.ndarray] = envs.success_evaluator(
                    total_infos=total_infos,
                    total_batch_list=total_batch_list,
                    episode_rewards=episode_rewards,
                    episode_lengths=episode_lengths,
                    )

        # Collect A/B mask from the environment (set during reset())
        wsm_traj = getattr(envs, "with_skills_mask", None)
        if wsm_traj is not None:
            wsm_traj = np.asarray(wsm_traj, dtype=bool).copy()

        return total_batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings, wsm_traj, traj_steps, traj_action_history, traj_admissible

    def dynamic_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            ) -> DataProto:
        """
        Conduct dynamic rollouts until a target batch size is met. 
        Keeps sampling until the desired number of effective trajectories is collected.
        Adopted from DAPO (https://arxiv.org/abs/2503.14476)

        Args:
            gen_batch (DataProto): Initial batch for rollout.
            actor_rollout_wg: Actor model workers for generating responses.
            envs (EnvironmentManagerBase): Environment manager instance.

        Returns:
            total_batch_list (List[Dict]): Complete set of rollout steps.
            total_episode_rewards (np.ndarray): Accumulated rewards.
            total_episode_lengths (np.ndarray): Lengths per episode.
            total_success (Dict[str, np.ndarray]): Success metrics.
            total_traj_uid (np.ndarray): Trajectory IDs.
        """
        total_batch_list = []
        total_episode_rewards = []
        total_episode_lengths = []
        total_success = []
        total_traj_uid = []
        total_tool_callings = []
        total_wsm_chunks = []
        try_count: int = 0
        max_try_count = self.config.algorithm.filter_groups.max_num_gen_batches

        while len(total_batch_list) < self.config.data.train_batch_size * self.config.env.rollout.n and try_count < max_try_count:

            if len(total_batch_list) > 0:
                print(f"valid num={len(total_batch_list)} < target num={self.config.data.train_batch_size * self.config.env.rollout.n}. Keep generating... ({try_count}/{max_try_count})")
            try_count += 1

            batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings, wsm_traj, _ = self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )

            # Compute keep_indices before filter_group_data so we can apply the
            # same mask to wsm_traj (filter_group_data does not handle it).
            if wsm_traj is not None and not (try_count == max_try_count):
                pre_filter_size = len(batch_list)
                batch_list_f, episode_rewards_f, episode_lengths_f, success_f, traj_uid_f, tool_callings_f = filter_group_data(
                    batch_list=batch_list,
                    episode_rewards=episode_rewards,
                    episode_lengths=episode_lengths,
                    success=success,
                    traj_uid=traj_uid,
                    tool_callings=tool_callings,
                    config=self.config,
                    last_try=False,
                )
                # Reconstruct which original traj indices were kept by matching traj_uid
                kept_uids = set(traj_uid_f.tolist())
                keep_mask = np.array([uid in kept_uids for uid in traj_uid.tolist()], dtype=bool)
                wsm_traj = wsm_traj[keep_mask] if len(keep_mask) == len(wsm_traj) else wsm_traj
                batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings = (
                    batch_list_f, episode_rewards_f, episode_lengths_f, success_f, traj_uid_f, tool_callings_f
                )
            else:
                batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings = filter_group_data(
                    batch_list=batch_list,
                    episode_rewards=episode_rewards,
                    episode_lengths=episode_lengths,
                    success=success,
                    traj_uid=traj_uid,
                    tool_callings=tool_callings,
                    config=self.config,
                    last_try=(try_count == max_try_count),
                )

            total_batch_list += batch_list
            total_episode_rewards.append(episode_rewards)
            total_episode_lengths.append(episode_lengths)
            total_success.append(success)
            total_traj_uid.append(traj_uid)
            total_tool_callings.append(tool_callings)
            if wsm_traj is not None:
                total_wsm_chunks.append(wsm_traj)

        total_episode_rewards = np.concatenate(total_episode_rewards, axis=0)
        total_episode_lengths = np.concatenate(total_episode_lengths, axis=0)
        total_success = {key: np.concatenate([success[key] for success in total_success], axis=0) for key in total_success[0].keys()}
        total_traj_uid = np.concatenate(total_traj_uid, axis=0)
        total_tool_callings = np.concatenate(total_tool_callings, axis=0)
        total_wsm = np.concatenate(total_wsm_chunks, axis=0) if total_wsm_chunks else None

        # dynamic path does not collect traj_steps (trajectories injected via vanilla calls)
        return total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, total_tool_callings, total_wsm, None, None

    def multi_turn_loop(
            self,
            gen_batch: DataProto,
            actor_rollout_wg,
            envs: EnvironmentManagerBase,
            is_train: bool = True,
            ) -> DataProto:
        """
        Select and run the appropriate rollout loop (dynamic or vanilla).

        A/B skill injection is handled transparently by the environment manager:
        envs.reset() sets with_skills_mask and clears retrieved_memories for the
        no-skill arm, so no special rollout branch is needed here.

        Args:
            gen_batch (DataProto): Initial prompt batch.
            actor_rollout_wg: Actor model workers.
            envs (EnvironmentManagerBase): Environment manager for interaction.
            is_train (bool): Whether in training mode (affects dynamic sampling).

        Returns:
            DataProto: Final collected trajectory data with metadata.
        """
        if is_train:
            gen_batch = gen_batch.repeat(repeat_times=self.config.env.rollout.n, interleave=True)

        total_traj_steps = None
        total_traj_action_history = None
        total_traj_admissible = None
        if self.config.algorithm.filter_groups.enable and is_train:
            # Dynamic Sampling (for DAPO and Dynamic GiGPO)
            # traj_admissible not collected in dynamic path (remains None)
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, totoal_tool_callings, total_wsm, total_traj_steps, total_traj_action_history = \
                self.dynamic_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
            total_traj_admissible = None  # dynamic path does not collect admissible cmds
        else:
            # Vanilla Sampling (A/B mask handled inside envs.reset())
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, totoal_tool_callings, total_wsm, total_traj_steps, total_traj_action_history, total_traj_admissible = \
                self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )

        assert len(total_batch_list) == len(total_episode_rewards)
        assert len(total_batch_list) == len(total_episode_lengths)
        assert len(total_batch_list) == len(total_traj_uid)
        assert len(total_batch_list) == len(totoal_tool_callings)

        gen_batch_output: DataProto = self.gather_rollout_data(
            total_batch_list=total_batch_list,
            episode_rewards=total_episode_rewards,
            episode_lengths=total_episode_lengths,
            success=total_success,
            traj_uid=total_traj_uid,
            tool_callings=totoal_tool_callings,
            with_skills_per_traj=total_wsm,
        )

        # Inject per-trajectory skill/task metadata for skill evolution.
        # traj_index is row-level (one entry per token row); use it to fan out
        # traj-level arrays from envs into row-level arrays in non_tensor_batch.
        traj_idx_arr = gen_batch_output.non_tensor_batch.get("traj_index")
        if traj_idx_arr is not None and hasattr(envs, 'tasks') and envs.tasks:
            traj_idx = np.asarray(traj_idx_arr, dtype=np.int64).ravel()
            n_trajs = len(envs.tasks)

            tasks_arr = np.array(envs.tasks, dtype=object)
            gen_batch_output.non_tensor_batch["task_description"] = tasks_arr[
                np.clip(traj_idx, 0, n_trajs - 1)
            ]

            task_types_arr = None
            if hasattr(envs, 'task_types') and envs.task_types:
                task_types_arr = np.array(envs.task_types, dtype=object)
            elif hasattr(envs, 'retrieval_memory') and envs.retrieval_memory is not None:
                # Derive task_type from memory retrieval results
                retrieved = getattr(envs, 'retrieved_memories', None) or []
                task_types_arr = np.array(
                    [retrieved[i].get('task_type', '') if i < len(retrieved) and retrieved[i] else ''
                     for i in range(n_trajs)],
                    dtype=object,
                )
            if task_types_arr is not None:
                gen_batch_output.non_tensor_batch["task_type"] = task_types_arr[
                    np.clip(traj_idx, 0, n_trajs - 1)
                ]

            retrieved = getattr(envs, 'retrieved_memories', None) or []
            skill_ids_per_traj = []
            for i in range(n_trajs):
                mem = retrieved[i] if i < len(retrieved) and retrieved[i] else {}
                ids = [s['skill_id'] for s in mem.get('general_skills', []) if 'skill_id' in s]
                ids += [s['skill_id'] for s in mem.get('task_specific_skills', []) if 'skill_id' in s]
                skill_ids_per_traj.append(ids)
            skill_ids_arr = np.array(skill_ids_per_traj, dtype=object)
            gen_batch_output.non_tensor_batch["skill_ids_used"] = skill_ids_arr[
                np.clip(traj_idx, 0, n_trajs - 1)
            ]

            # Inject action/observation trajectory steps (for skill failure analysis)
            if total_traj_steps is not None:
                steps_arr = np.array(total_traj_steps, dtype=object)  # (n_trajs,) list-of-dicts
                gen_batch_output.non_tensor_batch["trajectories"] = steps_arr[
                    np.clip(traj_idx, 0, n_trajs - 1)
                ]
            # Inject raw action history for replay-based skill verification
            if total_traj_action_history is not None:
                action_hist_arr = np.array(total_traj_action_history, dtype=object)
                gen_batch_output.non_tensor_batch["traj_action_history"] = action_hist_arr[
                    np.clip(traj_idx, 0, n_trajs - 1)
                ]
            # Inject per-step admissible commands for anchor buffer
            if total_traj_admissible is not None:
                adm_arr = np.array(total_traj_admissible, dtype=object)
                gen_batch_output.non_tensor_batch["traj_admissible"] = adm_arr[
                    np.clip(traj_idx, 0, n_trajs - 1)
                ]
            # Inject gamefile per trajectory (ALFWorld only)
            if hasattr(envs, 'gamefile') and envs.gamefile:
                gf_arr = np.array(envs.gamefile, dtype=object)
                gen_batch_output.non_tensor_batch["traj_gamefile"] = gf_arr[
                    np.clip(traj_idx, 0, n_trajs - 1)
                ]
            # Inject session_idx per trajectory (WebShop only)
            if hasattr(envs, 'session_idxs') and envs.session_idxs:
                si_arr = np.array(envs.session_idxs, dtype=object)
                gen_batch_output.non_tensor_batch["traj_session_idx"] = si_arr[
                    np.clip(traj_idx, 0, n_trajs - 1)
                ]
            # Inject task_kwargs per trajectory (Search only)
            if hasattr(envs, 'task_kwargs') and envs.task_kwargs:
                tk_arr = np.array(envs.task_kwargs, dtype=object)
                gen_batch_output.non_tensor_batch["traj_task_kwargs"] = tk_arr[
                    np.clip(traj_idx, 0, n_trajs - 1)
                ]

        # Surface global per-arm success rates in meta_info (only when A/B split exists)
        if total_wsm is not None:
            sr_skill = gen_batch_output.non_tensor_batch.get("success_rate_skill")
            sr_origin = gen_batch_output.non_tensor_batch.get("success_rate_origin")
            if sr_skill is not None:
                gen_batch_output.meta_info['skill_sr'] = float(np.mean(sr_skill))
            if sr_origin is not None:
                gen_batch_output.meta_info['noskill_sr'] = float(np.mean(sr_origin))

        # Per-task-type SR — computed regardless of A/B split.
        # When wsm is None (val_rollout_always_skills=True), all trajectories are
        # treated as skill arm so only skill_sr_{tt} is written (no noskill_sr_{tt}).
        task_type_nb = gen_batch_output.non_tensor_batch.get("task_type")
        traj_succ_nb = gen_batch_output.non_tensor_batch.get("traj_success")
        wsm_nb       = gen_batch_output.non_tensor_batch.get("with_skills_mask")
        traj_idx_nb  = gen_batch_output.non_tensor_batch.get("traj_index")
        if task_type_nb is not None and traj_succ_nb is not None and traj_idx_nb is not None:
            seen: set = set()
            tt_list, succ_list, wsm_list = [], [], []
            for i in range(len(traj_idx_nb)):
                tid = int(traj_idx_nb[i])
                if tid in seen:
                    continue
                seen.add(tid)
                tt_list.append(str(task_type_nb[i]))
                succ_list.append(float(bool(traj_succ_nb[i])))
                wsm_list.append(bool(wsm_nb[i]) if wsm_nb is not None else True)
            tt_arr   = np.array(tt_list)
            succ_arr = np.array(succ_list, dtype=np.float32)
            wsm_arr2 = np.array(wsm_list, dtype=bool)
            for tt in np.unique(tt_arr):
                tt_mask = tt_arr == tt
                skill_succ   = succ_arr[tt_mask & wsm_arr2]
                noskill_succ = succ_arr[tt_mask & ~wsm_arr2]
                if skill_succ.size > 0:
                    gen_batch_output.meta_info[f'skill_sr_{tt}'] = float(np.mean(skill_succ))
                if noskill_succ.size > 0:
                    gen_batch_output.meta_info[f'noskill_sr_{tt}'] = float(np.mean(noskill_succ))

        return gen_batch_output
