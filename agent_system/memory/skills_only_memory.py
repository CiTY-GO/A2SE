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

"""
Lightweight skills-only memory system.

This is a simplified version of RetrievalMemory that only uses Claude-style skills
without the overhead of loading and indexing trajectory memories.

Supports two retrieval modes:
  - "template": keyword-based task type detection + return all task-specific skills
    (original behaviour, zero latency, no GPU needed)
  - "embedding": encode the task description with Qwen3-Embedding-0.6B and rank
    both general and task-specific skills by cosine similarity, so only the
    top-k most relevant ones are injected into the prompt

Schema v3: each skill may contain a ``meta`` dict with keys
  D-PLAN, D-TRACK, D-EXEC, D-GUARD (include only relevant dims per skill)
and a ``synthesized_text`` field derived from those dimensions.
``utility`` and ``retrieval_count`` track skill quality over training.
"""

import copy
import json
import os
from typing import Dict, Any, List, Optional
from .base import BaseMemory


class SkillsOnlyMemory(BaseMemory):
    """
    Lightweight memory system that only uses Claude-style skills.

    Retrieval mode is controlled by the ``retrieval_mode`` constructor argument:

    * ``"template"`` (default) – keyword matching selects the task category;
      *all* task-specific skills for that category are returned, and the first
      ``top_k`` general skills are returned in document order.  No embedding
      model is needed.

    * ``"embedding"`` – the task description is encoded with a
      SentenceTransformer model (Qwen3-Embedding-0.6B by default).  Both
      general skills and task-specific skills (searched across **all**
      categories) are ranked by cosine similarity and the top-k are returned.
      Skill embeddings are pre-computed once and cached in memory.
    """

    # ------------------------------------------------------------------ #
    # Construction                                                         #
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        skills_json_path: str,
        retrieval_mode: str = "template",
        embedding_model_path: Optional[str] = None,
        task_specific_top_k: Optional[int] = None,
    ):
        """
        Args:
            skills_json_path:     Path to Claude-style skills JSON file.
            retrieval_mode:       ``"template"`` or ``"embedding"``.
            embedding_model_path: Local path (or HF model ID) for the
                                  SentenceTransformer embedding model.  Only
                                  used when ``retrieval_mode="embedding"``.
                                  Defaults to ``"Qwen/Qwen3-Embedding-0.6B"``.
            task_specific_top_k:  Maximum number of task-specific skills to
                                  return.  ``None`` means *return all* in
                                  template mode and use ``top_k`` (general
                                  skills count) in embedding mode.
        """
        if retrieval_mode not in ("template", "embedding"):
            raise ValueError(
                f"retrieval_mode must be 'template' or 'embedding', got '{retrieval_mode}'"
            )

        if not os.path.exists(skills_json_path):
            raise FileNotFoundError(f"Skills file not found: {skills_json_path}")

        with open(skills_json_path, 'r') as f:
            self.skills = json.load(f)

        self.retrieval_mode = retrieval_mode
        self.embedding_model_path = embedding_model_path or "/path/to/workspace/models/Qwen3-Embedding-0.6B"
        self.task_specific_top_k = task_specific_top_k

        # Lazy-initialised embedding state (only used in embedding mode)
        self._embedding_model = None
        self._skill_embeddings_cache: Optional[Dict] = None

        n_general = len(self.skills.get('general_skills', []))
        n_task = sum(len(v) for v in self.skills.get('task_specific_skills', {}).values())
        n_mistakes = len(self.skills.get('common_mistakes', []))
        print(
            f"[SkillsOnlyMemory] Loaded skills: {n_general} general, "
            f"{n_task} task-specific, {n_mistakes} mistakes  "
            f"| retrieval_mode={retrieval_mode}"
        )

        # In embedding mode, pre-compute skill embeddings eagerly so the first
        # retrieve() call is not slower than subsequent ones.
        if retrieval_mode == "embedding":
            self._compute_skill_embeddings()

    # ------------------------------------------------------------------ #
    # Task-type detection (template mode)                                  #
    # ------------------------------------------------------------------ #

    def _detect_task_type(self, task_description: str) -> str:
        """Detect task category. ALFWorld uses gamefile-based detection (via
        retrieve(task_type=...) parameter); this method only handles WebShop.
        For ALFWorld callers that lack a gamefile, returns 'pick_and_place' as
        a safe fallback and logs a warning.
        """
        task_specific = self.skills.get('task_specific_skills', {})
        goal = task_description.lower()

        # ---- ALFWorld fallback (should not be reached if gamefile is available) --
        if 'pick_and_place' in task_specific or 'clean' in task_specific:
            print(f"[SkillsOnlyMemory] WARNING: _detect_task_type called for ALFWorld "
                  f"without gamefile hint; falling back to 'pick_and_place'. "
                  f"Pass task_type= to retrieve() to suppress this warning.")
            return 'pick_and_place'

        # ---- WebShop categories -----------------------------------------
        elif 'apparel' in task_specific or 'electronics' in task_specific:
            if any(kw in goal for kw in [
                'shirt', 'dress', 'jacket', 'pant', 'coat', 'sweater',
                'blouse', 'clothing', 'clothes', 't-shirt',
            ]):
                return 'apparel'
            elif any(kw in goal for kw in [
                'shoe', 'boot', 'sneaker', 'sandal', 'heel', 'slipper',
                'footwear',
            ]):
                return 'footwear'
            elif any(kw in goal for kw in [
                'laptop', 'phone', 'computer', 'tablet', 'charger',
                'cable', 'headphone', 'speaker', 'camera', 'electronic',
            ]):
                return 'electronics'
            elif any(kw in goal for kw in [
                'necklace', 'ring', 'bracelet', 'earring', 'watch',
                'jewelry', 'bag', 'purse', 'wallet',
            ]):
                return 'accessories'
            elif any(kw in goal for kw in [
                'furniture', 'lamp', 'curtain', 'pillow', 'bedding',
                'decor', 'candle', 'vase', 'rug',
            ]):
                return 'home_decor'
            elif any(kw in goal for kw in [
                'cream', 'lotion', 'shampoo', 'conditioner', 'moisturizer',
                'serum', 'makeup', 'beauty', 'vitamin', 'supplement',
            ]):
                return 'beauty_health'
            else:
                return 'other'

        # ---- Fallback: first key in task_specific_skills, or 'unknown' --
        else:
            return next(iter(task_specific), 'unknown')

    # ------------------------------------------------------------------ #
    # Embedding helpers                                                    #
    # ------------------------------------------------------------------ #

    def _get_embedding_model(self):
        """Lazy-load the SentenceTransformer model (thread-safe for single process)."""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for embedding retrieval. "
                    "Install with: pip install sentence-transformers"
                )
            print(f"[SkillsOnlyMemory] Loading embedding model: {self.embedding_model_path}")
            self._embedding_model = SentenceTransformer(self.embedding_model_path)
            print("[SkillsOnlyMemory] Embedding model ready.")
        return self._embedding_model

    @staticmethod
    def _skill_to_text(skill: Dict[str, Any]) -> str:
        """Concatenate the skill fields most useful for semantic matching.

        Prefers ``synthesized_text`` (schema v2) over legacy ``principle``.
        """
        parts = []
        synth = skill.get('synthesized_text', '').strip()
        if synth:
            title = skill.get('title', '').strip()
            when = skill.get('when_to_apply', '').strip()
            if title:
                parts.append(title)
            parts.append(synth)
            if when:
                parts.append(when)
        else:
            for field in ('title', 'principle', 'when_to_apply'):
                val = skill.get(field, '').strip()
                if val:
                    parts.append(val)
        return ". ".join(parts)

    def _compute_skill_embeddings(self) -> Dict:
        """
        Pre-compute and cache normalised embeddings for every skill.

        The cache holds:
          ``items``      – flat list of ``(kind, task_type, skill_dict)``
          ``embeddings`` – numpy array of shape ``(n_skills, dim)``
          ``n_general``  – how many of the first rows correspond to general skills
        """
        if self._skill_embeddings_cache is not None:
            return self._skill_embeddings_cache

        import numpy as np

        general_items = [
            ('general', None, s)
            for s in self.skills.get('general_skills', [])
        ]
        task_items = [
            ('task_specific', task_type, s)
            for task_type, skills in self.skills.get('task_specific_skills', {}).items()
            for s in skills
        ]
        all_items = general_items + task_items
        texts = [self._skill_to_text(item[2]) for item in all_items]

        model = self._get_embedding_model()
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        self._skill_embeddings_cache = {
            'items': all_items,
            'embeddings': embeddings,
            'n_general': len(general_items),
        }
        print(
            f"[SkillsOnlyMemory] Cached embeddings for {len(all_items)} skills "
            f"({len(general_items)} general + {len(task_items)} task-specific)"
        )
        return self._skill_embeddings_cache

    def _embedding_retrieve(
        self,
        task_description: str,
        top_k_general: int,
        top_k_task_specific: int,
        task_type: str = None,
    ):
        """
        Retrieve the most relevant general and task-specific skills using
        cosine similarity between the task description and cached skill embeddings.

        Task-specific skills are restricted to the detected task_type category
        so skills from other categories are never returned.

        Args:
            task_description:    Free-form task goal string.
            top_k_general:       Number of general skills to return.
            top_k_task_specific: Number of task-specific skills to return
                                 (within the matched task_type only).
            task_type:           If provided, use this instead of auto-detection.

        Returns:
            Tuple of (general_skills, task_specific_skills).
        """
        import numpy as np

        cache = self._compute_skill_embeddings()
        model = self._get_embedding_model()

        query_emb = model.encode(
            [task_description],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )[0]  # shape: (dim,)

        sims = cache['embeddings'] @ query_emb  # cosine similarity, shape: (n,)

        n_general = cache['n_general']
        general_sims = sims[:n_general]

        # Top-k general skills
        general_idx = np.argsort(general_sims)[::-1][:top_k_general]
        general_skills = [cache['items'][int(i)][2] for i in general_idx]

        # Task-specific: restrict to the detected task_type category only.
        detected_type = task_type or self._detect_task_type(task_description)
        task_indices = [
            i for i, (kind, tt, _) in enumerate(cache['items'][n_general:])
            if tt == detected_type
        ]
        if task_indices:
            type_sims = np.array([sims[n_general + i] for i in task_indices])
            top_local = np.argsort(type_sims)[::-1][:top_k_task_specific]
            task_skills = [cache['items'][n_general + task_indices[int(j)]][2] for j in top_local]
        else:
            task_skills = []

        return general_skills, task_skills

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    def retrieve(
        self,
        task_description: str,
        top_k: int = 6,
        force_include_ids: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Retrieve skills for a given task description.

        Args:
            task_description:  Current task goal string.
            top_k:             Number of *general* skills to include.
                               In embedding mode this also serves as the
                               default for task-specific skills when
                               ``task_specific_top_k`` is not set.
            force_include_ids: Skill IDs that must appear in the result
                               regardless of retrieval ranking.  Used during
                               skill-change verification to guarantee the
                               updated/new skills are actually shown to the
                               agent.  These are prepended to the result and
                               do not count against ``top_k``.

        Returns:
            Dictionary with keys:
              - ``general_skills``       – list of skill dicts
              - ``task_specific_skills`` – list of skill dicts
              - ``mistakes_to_avoid``    – list of common-mistake dicts
              - ``task_type``            – detected task type string
              - ``task_specific_examples`` – always ``[]`` (reserved)
              - ``retrieval_mode``       – which mode was used
        """
        common_mistakes = []  # Migrated to skill D-GUARD fields; no longer injected into prompt

        # ----------------------------------------------------------------
        # Embedding mode: semantic ranking of all skills
        # ----------------------------------------------------------------
        # Caller may pass task_type directly (e.g. from gamefile) to bypass
        # keyword detection, which can be ambiguous for some task descriptions.
        task_type_hint = kwargs.get('task_type', None)

        if self.retrieval_mode == "embedding":
            ts_top_k = self.task_specific_top_k if self.task_specific_top_k is not None else top_k
            task_type = task_type_hint if task_type_hint else self._detect_task_type(task_description)
            general_skills, task_skills = self._embedding_retrieve(
                task_description=task_description,
                top_k_general=top_k,
                top_k_task_specific=ts_top_k,
                task_type=task_type,
            )
            if force_include_ids:
                general_skills, task_skills = self._apply_force_include(
                    force_include_ids, general_skills, task_skills
                )
            return {
                'general_skills': general_skills,
                'task_specific_skills': task_skills,
                'mistakes_to_avoid': common_mistakes,
                'task_type': task_type,
                'task_specific_examples': [],
                'retrieval_mode': 'embedding',
            }

        # ----------------------------------------------------------------
        # Template mode
        # ----------------------------------------------------------------
        task_type = task_type_hint if task_type_hint else self._detect_task_type(task_description)

        task_specific = self.skills.get('task_specific_skills', {})
        skill_key = task_type

        # Dynamic skills (dyn_NNN) are appended to the *end* of general_skills,
        # so a naive [:top_k] slice would silently drop all of them once the
        # static skill bank is larger than top_k.  Fix: always include every
        # dynamic skill, then fill the remaining budget with static skills.
        all_general = self.skills.get('general_skills', [])
        dynamic_skills = [s for s in all_general if s.get('skill_id', '').startswith('dyn_')]
        static_skills = [s for s in all_general if not s.get('skill_id', '').startswith('dyn_')]
        n_static = max(0, top_k - len(dynamic_skills))
        general_skills = dynamic_skills + static_skills[:n_static]

        all_task_skills = task_specific.get(skill_key, [])

        if self.task_specific_top_k is not None:
            task_skills = all_task_skills[:self.task_specific_top_k]
        else:
            task_skills = all_task_skills  # original behaviour: return all

        if force_include_ids:
            general_skills, task_skills = self._apply_force_include(
                force_include_ids, general_skills, task_skills
            )

        return {
            'general_skills': general_skills,
            'task_specific_skills': task_skills,
            'mistakes_to_avoid': common_mistakes,
            'task_type': task_type,
            'task_specific_examples': [],
            'retrieval_mode': 'template',
        }

    def format_for_prompt(self, retrieved_memories: Dict[str, Any]) -> str:
        """
        Format retrieved skills into a string suitable for prompt injection.

        Args:
            retrieved_memories: Dict returned by :meth:`retrieve`.

        Returns:
            Formatted multi-section string to insert into the agent prompt.
        """
        sections = []
        task_type = retrieved_memories.get('task_type', 'unknown')
        mode = retrieved_memories.get('retrieval_mode', 'template')

        # General skills
        general_skills = retrieved_memories.get('general_skills', [])
        if general_skills:
            lines = ["### General Principles"]
            for skill in general_skills:
                title = skill.get('title', '')
                text = skill.get('synthesized_text') or skill.get('principle', '')
                lines.append(f"- **{title}**: {text}")
            sections.append("\n".join(lines))

        # Task-specific skills
        task_skills = retrieved_memories.get('task_specific_skills', [])
        if task_skills:
            if mode == "embedding":
                section_title = "### Task-Relevant Skills"
            else:
                task_name = task_type.replace('_', ' ').title()
                section_title = f"### {task_name} Skills"
            lines = [section_title]
            for skill in task_skills:
                title = skill.get('title', '')
                text = skill.get('synthesized_text') or skill.get('principle', '')
                when = skill.get('when_to_apply', '')
                lines.append(f"- **{title}**: {text}")
                if when:
                    lines.append(f"  _Apply when: {when}_")
            sections.append("\n".join(lines))

        # Common mistakes
        mistakes = retrieved_memories.get('mistakes_to_avoid', [])
        if mistakes:
            lines = ["### Mistakes to Avoid"]
            for mistake in mistakes:
                desc = mistake.get('description', '')
                fix = mistake.get('how_to_avoid', '')
                if desc:
                    lines.append(f"- **Don't**: {desc}")
                    if fix:
                        lines.append(f"  **Instead**: {fix}")
            sections.append("\n".join(lines))

        return "\n\n".join(sections) if sections else "No relevant skills found for this task."

    # ------------------------------------------------------------------ #
    # BaseMemory interface (not used in skills-only memory)               #
    # ------------------------------------------------------------------ #

    def reset(self, batch_size: int):
        pass

    def store(self, record: Dict[str, List[Any]]):
        pass

    def fetch(self, step: int):
        pass

    def __len__(self):
        return (
            len(self.skills.get('general_skills', [])) +
            sum(len(v) for v in self.skills.get('task_specific_skills', {}).values()) +
            len(self.skills.get('common_mistakes', []))
        )

    def __getitem__(self, idx: int):
        return self.skills

    # ------------------------------------------------------------------ #
    # Dynamic update methods                                               #
    # ------------------------------------------------------------------ #

    def add_skills(self, new_skills: List[Dict], category: str = 'general') -> int:
        """
        Add new skills to the bank and invalidate the embedding cache.

        Args:
            new_skills: List of skill dicts to add.
            category:   ``'general'`` or a task-type key (e.g. ``'clean'``).

        Returns:
            Number of skills actually added (duplicates are skipped).
        """
        added = 0
        existing_ids = self._get_all_skill_ids()

        for skill in new_skills:
            skill_id = skill.get('skill_id')
            if skill_id in existing_ids:
                print(f"[SkillsOnlyMemory] Skipping duplicate skill: {skill_id}")
                continue

            if category == 'general':
                self.skills.setdefault('general_skills', []).append(skill)
            else:
                self.skills.setdefault('task_specific_skills', {}).setdefault(category, []).append(skill)
            added += 1
            print(f"[SkillsOnlyMemory] Added skill: {skill_id} - {skill.get('title', 'N/A')}")

        if added > 0:
            # Invalidate embedding cache so it is recomputed on next retrieve
            self._skill_embeddings_cache = None

        return added

    def remove_skill(self, skill_id: str) -> bool:
        """Remove a skill by ID and invalidate the embedding cache."""
        removed = False

        original_len = len(self.skills.get('general_skills', []))
        self.skills['general_skills'] = [
            s for s in self.skills.get('general_skills', [])
            if s.get('skill_id') != skill_id
        ]
        if len(self.skills.get('general_skills', [])) < original_len:
            removed = True

        for task_type in self.skills.get('task_specific_skills', {}):
            original_len = len(self.skills['task_specific_skills'][task_type])
            self.skills['task_specific_skills'][task_type] = [
                s for s in self.skills['task_specific_skills'][task_type]
                if s.get('skill_id') != skill_id
            ]
            if len(self.skills['task_specific_skills'][task_type]) < original_len:
                removed = True

        if removed:
            self._skill_embeddings_cache = None
            print(f"[SkillsOnlyMemory] Removed skill: {skill_id}")
        return removed

    def save_skills(self, path: str):
        """Persist the current skill bank to a JSON file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.skills, f, indent=2)
        print(f"[SkillsOnlyMemory] Saved {len(self)} skills to {path}")

    def _get_all_skill_ids(self) -> set:
        ids = set()
        for s in self.skills.get('general_skills', []):
            if s.get('skill_id'):
                ids.add(s['skill_id'])
        for task_skills in self.skills.get('task_specific_skills', {}).values():
            for s in task_skills:
                if s.get('skill_id'):
                    ids.add(s['skill_id'])
        return ids

    def get_skill_task_types(self, skill_ids: List[str]) -> set:
        """Return the set of task types that the given skill_ids belong to.

        General skills (``gen_``, ``dyn_`` in general_skills) map to the
        sentinel ``"general"``.  Task-specific skills map to their category
        key (e.g. ``"clean"``, ``"heat"``).  This is used by the verification
        rollout to select only tasks that actually exercise the changed skills.
        """
        result = set()
        general_ids = {s.get('skill_id') for s in self.skills.get('general_skills', [])}
        for sid in skill_ids:
            if sid in general_ids:
                result.add('general')
                continue
            for task_type, skills in self.skills.get('task_specific_skills', {}).items():
                if any(s.get('skill_id') == sid for s in skills):
                    result.add(task_type)
                    break
        return result

    def _apply_force_include(
        self,
        force_include_ids: List[str],
        general_skills: List[Dict],
        task_skills: List[Dict],
    ):
        """Prepend force-included skills to the result lists.

        Skills in ``force_include_ids`` that are not already present in
        ``general_skills`` or ``task_skills`` are looked up in the full skill
        bank and prepended.  This guarantees that updated/new skills are
        always visible to the agent during verification rollouts.
        """
        already_included = {s.get('skill_id') for s in general_skills + task_skills}
        to_prepend = []
        for sid in force_include_ids:
            if sid in already_included:
                continue
            skill = self._find_skill(sid)
            if skill is not None:
                to_prepend.append(skill)
        if to_prepend:
            general_skills = to_prepend + list(general_skills)
        return general_skills, task_skills

    def get_skill_count(self) -> Dict[str, int]:
        return {
            'general': len(self.skills.get('general_skills', [])),
            'task_specific': sum(len(v) for v in self.skills.get('task_specific_skills', {}).values()),
            'common_mistakes': len(self.skills.get('common_mistakes', [])),
            'total': len(self),
        }

    # ------------------------------------------------------------------ #
    # Schema-v2: meta-dimension updates & synthesized_text                #
    # ------------------------------------------------------------------ #

    META_DIMS = ('D-PLAN', 'D-TRACK', 'D-EXEC', 'D-GUARD')

    def _find_skill(self, skill_id: str) -> Optional[Dict]:
        """Return the mutable skill dict for ``skill_id``, or None."""
        for s in self.skills.get('general_skills', []):
            if s.get('skill_id') == skill_id:
                return s
        for task_skills in self.skills.get('task_specific_skills', {}).values():
            for s in task_skills:
                if s.get('skill_id') == skill_id:
                    return s
        return None

    def update_meta_dim(self, skill_id: str, dim: str, new_content: str) -> bool:
        """Update a single meta dimension for a skill.

        After calling this, call ``synthesize_skill`` or ``synthesize_all``
        to regenerate ``synthesized_text`` from the updated meta.

        Returns True if the skill was found and updated.
        """
        if dim not in self.META_DIMS:
            raise ValueError(f"dim must be one of {self.META_DIMS}, got '{dim}'")
        skill = self._find_skill(skill_id)
        if skill is None:
            print(f"[SkillsOnlyMemory] update_meta_dim: skill '{skill_id}' not found")
            return False
        skill.setdefault('meta', {})[dim] = new_content
        self._skill_embeddings_cache = None
        return True

    def synthesize_skill(self, skill: Dict, llm_client=None) -> str:
        """Generate ``synthesized_text`` from the skill's meta dimensions.

        If ``llm_client`` is provided (OpenAI-compatible), calls the LLM to
        produce a coherent paragraph.  Otherwise falls back to a simple
        rule-based concatenation.

        The result is written back into ``skill['synthesized_text']`` in-place
        and also returned.
        """
        meta = skill.get('meta', {})
        # 防御: LLM 有时返回 list of dict，统一转换为 dict
        if isinstance(meta, list):
            merged = {}
            for item in meta:
                if isinstance(item, dict):
                    merged.update(item)
            meta = merged
            skill['meta'] = meta
        if not meta:
            return skill.get('synthesized_text', skill.get('principle', ''))

        if llm_client is not None:
            from agent_system.memory.skill_updater import _llm_chat
            import os as _os
            _mis_id = _os.getenv("SKILLRL_MIS_ID") or _os.getenv("CATPAW_MIS_ID", "YOUR_MIS_ID")
            _model = (getattr(llm_client, 'model', None)
                      or getattr(llm_client, '_skill_model', None)
                      or _os.getenv('SKILLRL_LLM_MODEL', 'claude-sonnet-4-6'))
            _dim_desc = {
                'D-PLAN':  'Overall strategy decomposition and search/exploration heuristics',
                'D-TRACK': 'Continuous tracking of object states, positions, and task progress',
                'D-EXEC':  'Fine-grained action execution details and operation sequences',
                'D-GUARD': 'Explicit constraints against common failure modes and error patterns',
            }
            _dim_lines = "\n".join(
                f"  [{k}] ({_dim_desc.get(k, k)}): {v}"
                for k, v in meta.items() if v and k in self.META_DIMS
            )
            prompt = (
                "You are writing guidance for an embodied AI agent skill bank.\n\n"
                f"Skill: {skill.get('title', '')}\n"
                f"When to apply: {skill.get('when_to_apply', '')}\n\n"
                "The skill has the following meta dimensions (only those with content are shown):\n"
                f"{_dim_lines}\n\n"
                "Write a coherent paragraph (5-8 sentences) that integrates all provided dimensions "
                "into complete, actionable guidance. "
                "Cover strategy (D-PLAN: what to do first and how to search), "
                "tracking (D-TRACK: what state to monitor), "
                "execution (D-EXEC: exact action sequences), "
                "and constraints (D-GUARD: what never to do and common pitfalls to avoid). "
                "Write as direct imperative instructions to the agent. Return only the paragraph."
            )
            try:
                import re as _re
                raw = _llm_chat(
                    messages=[{"role": "user", "content": prompt}],
                    model=_model, mis_id=_mis_id, max_tokens=12000,
                )
                text = _re.sub(r'<think>.*?</think>', '', raw, flags=_re.DOTALL).strip()
                if text:
                    skill['synthesized_text'] = text
                    self._skill_embeddings_cache = None
                    return text
            except Exception as e:
                print(f"[SkillsOnlyMemory] synthesize_skill LLM error: {e}")

        # Fallback: rule-based concatenation
        parts = []
        for dim in self.META_DIMS:
            val = meta.get(dim, '').strip()
            if val:
                parts.append(val)
        text = ' '.join(parts)
        skill['synthesized_text'] = text
        self._skill_embeddings_cache = None
        return text

    def synthesize_all(self, llm_client=None, skill_ids=None):
        """Regenerate ``synthesized_text`` for skills that have a ``meta`` dict."""
        target = set(skill_ids) if skill_ids else None

        to_synthesize = []
        for s in self.skills.get('general_skills', []):
            if not s.get('meta'):
                continue
            if target is not None and s.get('skill_id') not in target:
                continue
            to_synthesize.append(s)
        for task_skills in self.skills.get('task_specific_skills', {}).values():
            for s in task_skills:
                if not s.get('meta'):
                    continue
                if target is not None and s.get('skill_id') not in target:
                    continue
                to_synthesize.append(s)

        if not to_synthesize:
            print(f"[SkillsOnlyMemory] synthesize_all: no skills to update")
            return

        for s in to_synthesize:
            self.synthesize_skill(s, llm_client=llm_client)

        self._skill_embeddings_cache = None
        print(f"[SkillsOnlyMemory] synthesize_all: updated {len(to_synthesize)} skills")


    # ------------------------------------------------------------------ #
    # Schema-v2: utility tracking                                          #
    # ------------------------------------------------------------------ #

    def update_utility(self, skill_ids: List[str], delta: float, ema_alpha: float = 0.3):
        """Apply an EMA utility update to a list of skills.

        ``delta > 0`` rewards skills that contributed to success;
        ``delta < 0`` penalises skills that appeared in failed episodes.
        """
        for skill_id in skill_ids:
            skill = self._find_skill(skill_id)
            if skill is None:
                continue
            old = skill.get('utility', 1.0)
            # EMA: new = alpha * (old + delta) + (1-alpha) * old  →  old + alpha*delta
            skill['utility'] = float(max(0.0, min(2.0, old + ema_alpha * delta)))
            skill['retrieval_count'] = skill.get('retrieval_count', 0) + 1

    def evict_low_utility(self, min_utility: float = 0.2) -> List[str]:
        """Remove skills whose utility has dropped below ``min_utility``.

        Dynamic skills (``dyn_`` prefix) are eligible for eviction; static
        skills are protected.

        Returns the list of evicted skill_ids.
        """
        evicted = []

        def _filter(skill_list):
            keep, removed = [], []
            for s in skill_list:
                sid = s.get('skill_id', '')
                if sid.startswith('dyn_') and s.get('utility', 1.0) < min_utility:
                    removed.append(sid)
                else:
                    keep.append(s)
            return keep, removed

        kept, rm = _filter(self.skills.get('general_skills', []))
        self.skills['general_skills'] = kept
        evicted.extend(rm)

        for task_type in list(self.skills.get('task_specific_skills', {}).keys()):
            kept, rm = _filter(self.skills['task_specific_skills'][task_type])
            self.skills['task_specific_skills'][task_type] = kept
            evicted.extend(rm)

        if evicted:
            self._skill_embeddings_cache = None
            print(f"[SkillsOnlyMemory] Evicted {len(evicted)} low-utility skills: {evicted}")
        return evicted

    # ------------------------------------------------------------------ #
    # Snapshot / restore for safe skill evolution                          #
    # ------------------------------------------------------------------ #

    def snapshot(self) -> Dict:
        """Return a deep copy of the current skill bank for rollback."""
        return copy.deepcopy(self.skills)

    def restore(self, snap: Dict):
        """Restore the skill bank from a previously taken snapshot."""
        self.skills = copy.deepcopy(snap)
        self._skill_embeddings_cache = None
        print("[SkillsOnlyMemory] Skill bank restored from snapshot.")

    def restore_partial(self, snap: Dict, skill_ids: List[str]):
        """Restore only the specified skill_ids to their snapshot state.

        Skills not in skill_ids are left at their current (updated) state.
        If a skill_id exists in snap, it is restored to that version.
        If a skill_id does not exist in snap (i.e. newly added), it is removed.
        """
        snap_map: Dict[str, Dict] = {}
        for s in snap.get('general_skills', []):
            snap_map[s['skill_id']] = ('general', s)
        for tt, skills in snap.get('task_specific_skills', {}).items():
            for s in skills:
                snap_map[s['skill_id']] = (tt, s)

        ids_to_restore = set(skill_ids)

        # Remove newly-added skills (not in snapshot)
        self.skills['general_skills'] = [
            s for s in self.skills.get('general_skills', [])
            if s['skill_id'] not in ids_to_restore or s['skill_id'] in snap_map
        ]
        for tt in list(self.skills.get('task_specific_skills', {}).keys()):
            self.skills['task_specific_skills'][tt] = [
                s for s in self.skills['task_specific_skills'][tt]
                if s['skill_id'] not in ids_to_restore or s['skill_id'] in snap_map
            ]

        # Restore modified or evicted skills to snapshot version
        for skill_id in ids_to_restore:
            if skill_id not in snap_map:
                continue  # already removed above (was newly added)
            category, snap_skill = snap_map[skill_id]
            if category == 'general':
                cur_ids = [s['skill_id'] for s in self.skills.get('general_skills', [])]
                if skill_id in cur_ids:
                    idx = cur_ids.index(skill_id)
                    self.skills['general_skills'][idx] = copy.deepcopy(snap_skill)
                else:
                    # skill was evicted (pruned) — re-insert it
                    self.skills.setdefault('general_skills', []).append(copy.deepcopy(snap_skill))
            else:
                ts = self.skills.setdefault('task_specific_skills', {}).setdefault(category, [])
                cur_ids = [s['skill_id'] for s in ts]
                if skill_id in cur_ids:
                    idx = cur_ids.index(skill_id)
                    ts[idx] = copy.deepcopy(snap_skill)
                else:
                    # skill was evicted (pruned) — re-insert it
                    ts.append(copy.deepcopy(snap_skill))

        self._skill_embeddings_cache = None
        print(f"[SkillsOnlyMemory] Partial restore: {len(ids_to_restore)} skill(s) rolled back.")
