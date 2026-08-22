"""
Anchor Buffer for skill-change verification in SkillRL.

Stores per-trajectory anchor records: (obs_prompt, admissible_commands, action_taken, success).
After a skill update, re-scores the stored prompts with the new skill text and looks up
whether the newly preferred action is historically better → provides a fast, env-free
estimate of whether the skill update helps.

Design principles:
- No env reset needed: uses stored (prompt, admissible_commands, success) triples
- Works with ALFWorld's discrete admissible action set
- Lightweight: only stores skill-arm trajectories, max_trajs entries
- Verifies via: new_skill_prompt → model top-action → lookup historical action success
"""

import numpy as np
from collections import defaultdict
from typing import List, Dict, Optional, Tuple, Any


class AnchorRecord:
    """One step from a skill-arm trajectory."""
    __slots__ = [
        'prompt',            # full prompt text at this step (with old skill)
        'admissible',        # list[str] of admissible actions at this step
        'action_taken',      # str: action the model actually took
        'step_success',      # float: episode success (1.0/0.0), assigned at episode end
        'task_type',         # str
        'gamefile',          # str (ALFWorld only)
        'traj_index',        # int
        'step_index',        # int: position in trajectory
    ]

    def __init__(self, prompt, admissible, action_taken, step_success,
                 task_type='', gamefile=None, traj_index=0, step_index=0):
        self.prompt = prompt
        self.admissible = list(admissible) if admissible else []
        self.action_taken = action_taken
        self.step_success = float(step_success) if step_success is not None else 0.0
        self.task_type = task_type
        self.gamefile = gamefile
        self.traj_index = traj_index
        self.step_index = step_index


class SkillAnchorBuffer:
    """
    Stores anchor records from skill-arm trajectories for skill verification.

    Verification flow:
      1. update_from_batch(batch)  — called each training step
      2. verify_skill_change(actor, skill_memory, changed_ids) — called after skill update
         a. For each record: rebuild prompt with new skills
         b. Ask actor for top-1 action from admissible set (single forward pass)
         c. Compare: did new top-action succeed historically more than old action?
         d. Return estimated post-SR delta

    Buffer keeps at most `max_trajs` unique trajectories, cycling out the oldest.
    """

    def __init__(self, max_trajs: int = 64, max_steps_per_traj: int = 10):
        self.max_trajs = max_trajs
        self.max_steps_per_traj = max_steps_per_traj
        # traj_key -> list[AnchorRecord]
        self._trajs: Dict[str, List[AnchorRecord]] = {}
        self._insertion_order: List[str] = []

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def update_from_batch(self, non_tensor_batch: dict) -> int:
        """
        Extract skill-arm step records from a training batch.

        Requires keys in non_tensor_batch:
          - with_skills_mask     (bool per row)
          - traj_index           (int per row)
          - traj_success         (float per row)
          - trajectories         (list[{action, observation}] per row)
          - traj_admissible      (list[list[str]] per row) — step-level admissible cmds
          - task_type            (str per row)
          - traj_gamefile        (str per row, optional)

        Returns number of new records added.
        """
        wsm = non_tensor_batch.get('with_skills_mask')
        traj_idx_arr = non_tensor_batch.get('traj_index')
        success_arr = non_tensor_batch.get('traj_success')
        traj_steps_arr = non_tensor_batch.get('trajectories')
        adm_arr = non_tensor_batch.get('traj_admissible')
        task_type_arr = non_tensor_batch.get('task_type')
        gamefile_arr = non_tensor_batch.get('traj_gamefile')

        if wsm is None or traj_idx_arr is None or traj_steps_arr is None:
            return 0

        # Debug: report whether admissible commands are available
        has_adm = adm_arr is not None and any(
            len(adm_arr[i]) > 0 for i in range(len(adm_arr))
            if adm_arr[i] is not None
        ) if adm_arr is not None else False
        if not has_adm:
            print(f"[AnchorBuffer] traj_admissible missing or empty — "
                  f"admissible_commands not collected in rollout_loop")

        added = 0
        seen_traj: set = set()
        for i in range(len(wsm)):
            if not bool(wsm[i]):
                continue  # no-skill arm, skip
            tid = int(traj_idx_arr[i])
            if tid in seen_traj:
                continue
            seen_traj.add(tid)

            steps = traj_steps_arr[i]  # list[{action, observation}]
            adm_steps = adm_arr[i] if adm_arr is not None else None  # list[list[str]]
            success = float(success_arr[i]) if success_arr is not None else 0.0
            task_type = str(task_type_arr[i]) if task_type_arr is not None else ''
            gamefile = str(gamefile_arr[i]) if gamefile_arr is not None else None

            if not steps:
                continue

            traj_key = f"{gamefile or tid}_{tid}"
            records = []
            for s_idx, step in enumerate(steps[:self.max_steps_per_traj]):
                adm = list(adm_steps[s_idx]) if adm_steps and s_idx < len(adm_steps) else []
                if not adm:
                    continue  # no admissible commands, skip this step
                records.append(AnchorRecord(
                    prompt=step.get('observation', ''),
                    admissible=adm,
                    action_taken=step.get('action', ''),
                    step_success=success,
                    task_type=task_type,
                    gamefile=gamefile,
                    traj_index=tid,
                    step_index=s_idx,
                ))

            if records:
                self._add_traj(traj_key, records)
                added += len(records)

        return added

    def _add_traj(self, key: str, records: List[AnchorRecord]):
        if key in self._trajs:
            self._trajs[key] = records  # overwrite with fresh data
        else:
            if len(self._trajs) >= self.max_trajs:
                oldest = self._insertion_order.pop(0)
                del self._trajs[oldest]
            self._trajs[key] = records
            self._insertion_order.append(key)

    # ------------------------------------------------------------------
    # Verify
    # ------------------------------------------------------------------

    def verify_skill_change(
        self,
        actor_rollout_wg,
        skill_memory,
        changed_skill_ids: List[str],
        retrieval_fn,
        rebuild_prompt_fn,
        tokenizer,
        n_samples: int = 32,
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Estimate pre/post SR using stored anchor records.

        Steps:
        1. Sample up to n_samples records from trajectories that used changed skills
        2. For each record:
           a. pre_success = record.step_success (historical)
           b. Rebuild prompt with new skill text → new_prompt
           c. Get model's top-1 action from admissible set (argmax log_prob)
           d. post_success = record.step_success if new_top_action == old_action else
                             estimate from other records with same gamefile
        3. Return (pre_sr, post_sr)

        Returns (None, None) if buffer is empty or not enough data.
        """
        if not self._trajs:
            return None, None

        # Collect all records
        all_records: List[AnchorRecord] = []
        for records in self._trajs.values():
            all_records.extend(records)

        if not all_records:
            return None, None

        # Sample
        if len(all_records) > n_samples:
            idxs = np.random.choice(len(all_records), n_samples, replace=False)
            sampled = [all_records[i] for i in idxs]
        else:
            sampled = all_records

        pre_successes = [r.step_success for r in sampled]
        pre_sr = float(np.mean(pre_successes)) if pre_successes else None

        n_same = 0      # new best == old action
        n_changed = 0   # new best != old action
        n_hist_found = 0
        n_hist_miss = 0
        n_no_adm = 0
        n_score_fail = 0
        sample_details = []  # collect up to 5 for logging

        # Rebuild prompts with new skills and score actions
        post_successes = []
        for record in sampled:
            if not record.admissible:
                post_successes.append(record.step_success)
                n_no_adm += 1
                continue
            try:
                new_prompt = rebuild_prompt_fn(
                    obs=record.prompt,
                    task_type=record.task_type,
                    skill_memory=skill_memory,
                )
                action_scores = _score_actions(
                    actor_rollout_wg=actor_rollout_wg,
                    prompt=new_prompt,
                    actions=record.admissible,
                    tokenizer=tokenizer,
                )
                if action_scores:
                    best_action = max(action_scores, key=action_scores.get)
                    best_score = action_scores[best_action]
                    old_score = action_scores.get(record.action_taken, float('nan'))
                    if best_action == record.action_taken:
                        post_successes.append(record.step_success)
                        n_same += 1
                        detail_tag = 'same'
                        post_succ = record.step_success
                    else:
                        hist = _lookup_action_success(
                            self._trajs, record.gamefile, best_action)
                        if hist is not None:
                            post_successes.append(hist)
                            n_hist_found += 1
                            detail_tag = f'changed(hist={hist:.2f})'
                            post_succ = hist
                        else:
                            post_successes.append(record.step_success)
                            n_hist_miss += 1
                            detail_tag = 'changed(hist=None,neutral)'
                            post_succ = record.step_success
                        n_changed += 1
                    if len(sample_details) < 5:
                        sample_details.append(
                            f"    traj={record.traj_index} step={record.step_index} "
                            f"type={record.task_type} pre_succ={record.step_success:.1f} "
                            f"post_succ={post_succ:.2f} [{detail_tag}]\n"
                            f"      old_action='{record.action_taken[:60]}' (score={old_score:.3f})\n"
                            f"      new_best  ='{best_action[:60]}' (score={best_score:.3f})\n"
                            f"      n_admissible={len(record.admissible)}"
                        )
                else:
                    post_successes.append(record.step_success)
                    n_score_fail += 1
            except Exception as e:
                print(f"[AnchorBuffer] verify step failed: {e}")
                post_successes.append(record.step_success)
                n_score_fail += 1

        post_sr = float(np.mean(post_successes)) if post_successes else None
        delta = (post_sr - pre_sr) if (pre_sr is not None and post_sr is not None) else float('nan')
        print(f"[SkillVerify/Anchor] sampled={len(sampled)} records: "
              f"same_action={n_same}, changed={n_changed} "
              f"(hist_found={n_hist_found}, hist_miss={n_hist_miss}), "
              f"no_admissible={n_no_adm}, score_fail={n_score_fail}")
        print(f"[SkillVerify/Anchor] pre_sr={pre_sr:.3f}, post_sr={post_sr:.3f}, "
              f"delta={delta:+.3f}")
        if sample_details:
            print(f"[SkillVerify/Anchor] sample details (first {len(sample_details)}):")
            for d in sample_details:
                print(d)
        return pre_sr, post_sr

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        n_trajs = len(self._trajs)
        n_records = sum(len(v) for v in self._trajs.values())
        task_types = defaultdict(int)
        for records in self._trajs.values():
            for r in records:
                task_types[r.task_type] += 1
        return {
            'n_trajs': n_trajs,
            'n_records': n_records,
            'task_type_counts': dict(task_types),
        }


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _score_actions(actor_rollout_wg, prompt: str, actions: List[str],
                   tokenizer) -> Dict[str, float]:
    """
    Score each admissible action by computing log P(action | prompt) via
    a single batched forward pass through the actor.

    Returns dict {action_text: log_prob}.
    """
    if not actions or actor_rollout_wg is None:
        return {}
    try:
        import torch
        scores = {}
        # Encode prompt once
        prompt_ids = tokenizer.encode(prompt, return_tensors='pt')
        for action in actions:
            action_ids = tokenizer.encode(
                action, add_special_tokens=False, return_tensors='pt')
            # Concatenate and get log-prob of action tokens
            input_ids = torch.cat([prompt_ids, action_ids], dim=1)
            with torch.no_grad():
                # Use actor_rollout_wg's compute_log_probs if available
                if hasattr(actor_rollout_wg, 'compute_log_probs'):
                    lp = actor_rollout_wg.compute_log_probs(input_ids)
                    # Sum log-probs over action tokens only
                    n_action = action_ids.shape[1]
                    scores[action] = float(lp[0, -n_action:].sum())
                else:
                    # Fallback: uniform (can't score without access to logits)
                    scores[action] = 0.0
        return scores
    except Exception:
        return {}


def _lookup_action_success(trajs: Dict, gamefile: Optional[str],
                           action: str) -> Optional[float]:
    """Look up mean success of `action` in records with the same gamefile."""
    successes = []
    for records in trajs.values():
        for r in records:
            if gamefile and r.gamefile != gamefile:
                continue
            if r.action_taken == action:
                successes.append(r.step_success)
    return float(np.mean(successes)) if successes else None
