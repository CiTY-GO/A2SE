"""
SkillPruner: success-driven skill simplification (less is better).

Workflow:
  1. ``attribute_from_successes``  – count which skill_ids appeared in
     successful trajectories.
  2. ``prune``                     – lower utility for un-attributed skills,
     evict below-threshold skills, merge near-duplicates.
  3. Caller runs a small validation rollout; if success_rate drops, it
     calls ``memory.restore(snapshot)`` to revert.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from agent_system.memory.skills_only_memory import SkillsOnlyMemory


class SkillPruner:
    def __init__(
        self,
        utility_reward: float = 0.2,
        utility_penalty: float = 0.1,
        min_utility: float = 0.2,
        merge_threshold: float = 0.85,
    ):
        """
        Args:
            utility_reward:   Utility delta applied to skills that appear in
                              successful trajectories.
            utility_penalty:  Utility delta applied to skills that do NOT
                              appear in any successful trajectory in this round.
            min_utility:      Threshold below which dynamic skills are evicted.
            merge_threshold:  Cosine similarity threshold for merging near-
                              duplicate skills.
        """
        self.utility_reward = utility_reward
        self.utility_penalty = utility_penalty
        self.min_utility = min_utility
        self.merge_threshold = merge_threshold

    # ------------------------------------------------------------------ #
    # Step 1: attribution                                                  #
    # ------------------------------------------------------------------ #

    def attribute_from_successes(
        self,
        success_trajs: List[Dict],
    ) -> Dict[str, int]:
        """Count how many successful trajectories each skill_id appeared in.

        Args:
            success_trajs: List of trajectory dicts.  Each dict may contain
                           ``skill_ids_used`` (list[str]) populated by the
                           rollout loop.

        Returns:
            ``{skill_id: count}`` mapping.
        """
        counter: Counter = Counter()
        for traj in success_trajs:
            for sid in traj.get('skill_ids_used', []):
                counter[sid] += 1
        return dict(counter)

    # ------------------------------------------------------------------ #
    # Step 2: prune                                                        #
    # ------------------------------------------------------------------ #

    def prune(
        self,
        memory: "SkillsOnlyMemory",
        attribution: Dict[str, int],
        all_skill_ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Apply utility updates and evict low-utility dynamic skills.

        Args:
            memory:        Live SkillsOnlyMemory instance.
            attribution:   Output of ``attribute_from_successes``.
            all_skill_ids: Full list of skill_ids that were retrieved during
                           this round (both skill and no-skill arms).  If
                           provided, skills that were retrieved but not
                           attributed get a penalty; others are untouched.

        Returns:
            List of evicted skill_ids.
        """
        attributed = set(attribution.keys())

        if all_skill_ids is not None:
            retrieved = set(all_skill_ids)
            # Reward attributed skills
            if attributed:
                memory.update_utility(list(attributed), delta=self.utility_reward)
            # Penalise retrieved-but-not-attributed skills
            penalised = list(retrieved - attributed)
            if penalised:
                memory.update_utility(penalised, delta=-self.utility_penalty)
        else:
            # Without retrieval list, only reward attributed skills
            if attributed:
                memory.update_utility(list(attributed), delta=self.utility_reward)

        # Merge near-duplicates before eviction
        merged = self.merge_similar(memory)

        # Evict low-utility dynamic skills
        evicted = memory.evict_low_utility(min_utility=self.min_utility)
        return evicted + [sid for sid in merged if sid not in evicted]

    # ------------------------------------------------------------------ #
    # Step 3: merge near-duplicates                                        #
    # ------------------------------------------------------------------ #

    def merge_similar(self, memory: "SkillsOnlyMemory") -> List[str]:
        """Merge pairs of skills whose ``synthesized_text`` embeddings are
        above ``merge_threshold`` cosine similarity.

        The skill with *lower* utility is removed; the higher-utility skill
        is kept unchanged.  Only dynamic skills (``dyn_`` prefix) are
        eligible for removal to protect the static skill bank.

        Skipped when ``retrieval_mode="template"`` since no embedding model
        is available in that mode.

        Returns:
            List of removed skill_ids.
        """
        # embedding model is only available in embedding retrieval mode
        if getattr(memory, 'retrieval_mode', 'template') != 'embedding':
            return []

        try:
            import numpy as np
        except ImportError:
            return []

        # Collect all skills with their text for embedding
        all_skills = []
        for s in memory.skills.get('general_skills', []):
            all_skills.append(s)
        for task_skills in memory.skills.get('task_specific_skills', {}).values():
            for s in task_skills:
                all_skills.append(s)

        if len(all_skills) < 2:
            return []

        texts = [
            s.get('synthesized_text') or s.get('principle', '') or s.get('title', '')
            for s in all_skills
        ]

        try:
            model = memory._get_embedding_model()
            embeddings = model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        except Exception as e:
            print(f"[SkillPruner] merge_similar embedding error: {e}")
            return []

        sims = embeddings @ embeddings.T  # (n, n)
        removed_ids: list = []

        for i in range(len(all_skills)):
            id_i = all_skills[i].get('skill_id', '')
            if id_i in removed_ids:
                continue  # already removed in a previous iteration
            for j in range(i + 1, len(all_skills)):
                id_j = all_skills[j].get('skill_id', '')
                if id_j in removed_ids:
                    continue  # already removed
                if sims[i, j] < self.merge_threshold:
                    continue
                si, sj = all_skills[i], all_skills[j]
                u_i = si.get('utility', 1.0)
                u_j = sj.get('utility', 1.0)
                # Only remove dynamic skills; keep the higher-utility one
                if id_j.startswith('dyn_') and u_i >= u_j:
                    memory.remove_skill(id_j)
                    removed_ids.append(id_j)
                    print(f"[SkillPruner] Merged '{id_j}' into '{id_i}' (sim={sims[i,j]:.3f})")
                elif id_i.startswith('dyn_') and u_j > u_i:
                    memory.remove_skill(id_i)
                    removed_ids.append(id_i)
                    print(f"[SkillPruner] Merged '{id_i}' into '{id_j}' (sim={sims[i,j]:.3f})")
                    break  # id_i is gone; skip remaining j for this i

        return removed_ids
