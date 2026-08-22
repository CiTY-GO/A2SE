"""
LLM-based skill updater that generates new skills from failed trajectories.

Supported backends (configured via SKILLRL_LLM_BACKEND env var):
  catpaw   – Claude via SKILLRL_CATPAW_BASE_URL  (Anthropic Messages API)
             Auth: SKILLRL_MIS_ID (Bearer token + x-api-key)
  aigc     – AIGC gateway via SKILLRL_AIGC_BASE_URL (OpenAI-compatible)
             Auth: SKILLRL_AIGC_APP_ID (Bearer token)
             Models: minimax / gemini-3.1-pro-preview (set SKILLRL_LLM_MODEL)

Default backend: catpaw
Default model  : claude-sonnet-4-6  (catpaw) / minimax (aigc)

AppId mapping:
  minimax               → SKILLRL_AIGC_APP_ID_MINIMAX
  gemini-3.1-pro-preview → SKILLRL_AIGC_APP_ID_GEMINI
"""
import json
import os
import re
from typing import List, Dict, Any, Optional

import httpx

# ── CatPaw (Claude / Anthropic Messages API) ──────────────────────────────────
_CATPAW_BASE_URL = os.getenv("SKILLRL_CATPAW_BASE_URL", "https://YOUR_LLM_BASE_URL")
_CATPAW_API_PATH = "/v1/messages"
_CLAUDE_CODE_SYSTEM = "You are Claude Code, Anthropic's official CLI for Claude."

# ── AIGC gateway (OpenAI-compatible) ─────────────────────────────────────────
_AIGC_BASE_URL = os.getenv("SKILLRL_AIGC_BASE_URL", "https://YOUR_AIGC_BASE_URL")
_AIGC_API_PATH = "/v1/openai/native/chat/completions"

# AppId per model (can be overridden by env vars)
_AIGC_APP_IDS = {
    "MiniMax-M2.7":           os.getenv("SKILLRL_AIGC_APP_ID_MINIMAX", "YOUR_AIGC_APP_ID_MINIMAX"),
    "gemini-3.1-pro-preview": os.getenv("SKILLRL_AIGC_APP_ID_GEMINI",  "YOUR_AIGC_APP_ID_GEMINI"),
}


def _make_catpaw_headers(mis_id: str) -> dict:
    return {
        "Authorization":  f"Bearer {mis_id}",
        "x-api-key":      f"sk-ant-oat01-{mis_id}",
        "x-ide-type":     "CatPaw_IDE",
        "x-working-dir":  "unknown",
        "x-repo-url":     "unknown",
        "x-branch":       "unknown",
        "x-app":          "cli",
        "user-agent":     "claude-cli/1.0.0",
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "claude-code-20250219,oauth-2025-04-20",
        "anthropic-dangerous-direct-browser-access": "true",
        "content-type":   "application/json",
    }


def _catpaw_chat(messages: list, model: str, mis_id: str,
                 max_tokens: int = 8192, timeout: int = 600,
                 max_retries: int = 3) -> str:
    """Call CatPaw via direct httpx POST to SKILLRL_CATPAW_BASE_URL, return response text."""
    prompt = "\n\n".join(
        m["content"] for m in messages if m.get("role") == "user"
    )
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{
            "role": "user",
            "content": [{"type": "text", "text": prompt,
                         "cache_control": {"type": "ephemeral"}}]
        }],
        "system": [{"type": "text", "text": _CLAUDE_CODE_SYSTEM,
                    "cache_control": {"type": "ephemeral"}}],
        "stream": False,
    }
    url = _CATPAW_BASE_URL + _CATPAW_API_PATH
    headers = _make_catpaw_headers(mis_id)

    for attempt in range(max_retries):
        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
            for block in data.get("content", []):
                if block.get("type") == "text":
                    return block["text"]
            return ""
        except httpx.HTTPStatusError as e:
            err = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
            if attempt == max_retries - 1:
                raise RuntimeError(f"[CatPaw] {err}")
            wait = 2 ** attempt
            print(f"[CatPaw] {err}, retrying in {wait}s ({attempt+1}/{max_retries})")
            import time; time.sleep(wait)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt
            print(f"[CatPaw] error: {e}, retrying in {wait}s ({attempt+1}/{max_retries})")
            import time; time.sleep(wait)
    raise RuntimeError(f"[CatPaw] Failed after {max_retries} retries")


def _aigc_chat(messages: list, model: str, app_id: str,
               max_tokens: int = 8192, timeout: int = 600,
               max_retries: int = 3) -> str:
    """Call AIGC gateway (OpenAI-compatible) via SKILLRL_AIGC_BASE_URL, return response text."""
    url = _AIGC_BASE_URL + _AIGC_API_PATH
    headers = {
        "Authorization": f"Bearer {app_id}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model":      model,
        "messages":   messages,
        "stream":     False,
        "max_tokens": max_tokens,
    }

    for attempt in range(max_retries):
        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
            # OpenAI-compatible response: choices[0].message.content
            choices = data.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "")
            return ""
        except httpx.HTTPStatusError as e:
            err = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
            if attempt == max_retries - 1:
                raise RuntimeError(f"[AIGC] {err}")
            wait = 2 ** attempt
            print(f"[AIGC] {err}, retrying in {wait}s ({attempt+1}/{max_retries})")
            import time; time.sleep(wait)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt
            print(f"[AIGC] error: {e}, retrying in {wait}s ({attempt+1}/{max_retries})")
            import time; time.sleep(wait)
    raise RuntimeError(f"[AIGC] Failed after {max_retries} retries")


def _llm_chat(messages: list, model: str, mis_id: str,
              max_tokens: int = 8192, timeout: int = 600,
              max_retries: int = 3) -> str:
    """Unified LLM call: routes to CatPaw or AIGC based on SKILLRL_LLM_BACKEND."""
    backend = os.getenv("SKILLRL_LLM_BACKEND", "catpaw").lower()
    if backend == "aigc":
        app_id = _AIGC_APP_IDS.get(model, os.getenv("SKILLRL_AIGC_APP_ID_MINIMAX", "1996479198143275053"))
        print(f"[LLM] backend=aigc model={model} app_id={app_id}")
        return _aigc_chat(messages, model, app_id,
                          max_tokens=max_tokens, timeout=timeout, max_retries=max_retries)
    else:
        print(f"[LLM] backend=catpaw model={model} mis_id={mis_id}")
        return _catpaw_chat(messages, model, mis_id,
                            max_tokens=max_tokens, timeout=timeout, max_retries=max_retries)




class SkillUpdater:
    def __init__(
        self,
        max_new_skills_per_update: int = 3,
        max_completion_tokens: int = 8192,
        model: str = "claude-sonnet-4-6",
    ):
        # SKILLRL_MIS_ID is set in config.sh; fall back to legacy CATPAW_MIS_ID, then hardcoded default
        self._mis_id = os.getenv("SKILLRL_MIS_ID") or os.getenv("CATPAW_MIS_ID", "YOUR_MIS_ID")
        self.model = model
        self.max_completion_tokens = max_completion_tokens
        self.max_new_skills_per_update = max_new_skills_per_update
        self.update_history = []
        # keep a dummy .client attribute so ray_trainer's _skill_model tag still works
        self.client = type("_stub", (), {"_skill_model": model})()

    def analyze_failures(
        self,
        failed_trajectories: List[Dict],
        current_skills: Dict,
    ) -> List[Dict]:
        """
        Analyse failed trajectories and generate new skills to address the gaps.

        Args:
            failed_trajectories: List of dicts with keys:
                ``task``       – task description string
                ``trajectory`` – list of ``{action, observation}`` step dicts
                ``task_type``  – detected task category string
            current_skills: The current skill bank dict (with keys
                ``general_skills``, ``task_specific_skills``, etc.)

        Returns:
            List of new skill dicts ready to be passed to
            ``SkillsOnlyMemory.add_skills()``.
        """
        if not failed_trajectories:
            return []

        # Compute the next available dyn_ index BEFORE calling the LLM so we
        # can tell it which IDs to use, avoiding duplicate-ID collisions.
        next_dyn_idx = self._next_dyn_index(current_skills)

        prompt = self._build_analysis_prompt(
            failed_trajectories, current_skills, next_dyn_idx
        )

        print(f"[SkillUpdater] analyze_failures {len(failed_trajectories)} failures prompt_len={len(prompt)}")
        try:
            content = _llm_chat(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                mis_id=self._mis_id,
            )
            if not content:
                print(f"[SkillUpdater] Empty response content, skipping")
                return []
            raw_skills = self._parse_skills_response(content)

            # Reassign dyn_ IDs on our side to guarantee no collisions,
            # regardless of what the LLM returned.
            reassigned = self._reassign_dyn_ids(raw_skills, next_dyn_idx)

            self.update_history.append({
                'num_failures_analyzed': len(failed_trajectories),
                'num_skills_generated': len(reassigned),
                'skill_ids': [s.get('skill_id') for s in reassigned],
            })

            return reassigned[:self.max_new_skills_per_update]

        except Exception as e:
            print(f"[SkillUpdater] Error calling {self.model}: {e}")
            return []

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _next_dyn_index(self, current_skills: Dict) -> int:
        """
        Scan the current skill bank for existing ``dyn_NNN`` IDs and return
        the next unused integer index (1-based).
        """
        max_idx = 0
        pattern = re.compile(r'^dyn_(\d+)$')

        for skill in current_skills.get('general_skills', []):
            m = pattern.match(skill.get('skill_id', ''))
            if m:
                max_idx = max(max_idx, int(m.group(1)))

        for skills in current_skills.get('task_specific_skills', {}).values():
            for skill in skills:
                m = pattern.match(skill.get('skill_id', ''))
                if m:
                    max_idx = max(max_idx, int(m.group(1)))

        return max_idx + 1

    def _reassign_dyn_ids(self, skills: List[Dict], start_idx: int) -> List[Dict]:
        """
        Replace whatever skill_id values the LLM returned with guaranteed-unique
        ``dyn_NNN`` IDs starting from ``start_idx``.
        """
        reassigned = []
        for i, skill in enumerate(skills):
            updated = dict(skill)
            updated['skill_id'] = f"dyn_{start_idx + i:03d}"
            reassigned.append(updated)
        return reassigned

    def _build_analysis_prompt(
        self,
        failed_trajectories: List[Dict],
        current_skills: Dict,
        next_dyn_idx: int,
    ) -> str:
        # Format failure examples
        failure_examples = []
        for i, traj in enumerate(failed_trajectories[:5]):
            failure_examples.append(
                f"\nExample {i + 1}:\n"
                f"Task: {traj['task']}\n"
                f"Task Type: {traj['task_type']}\n"
                f"Trajectory (last 5 steps):\n"
                f"{self._format_trajectory(traj['trajectory'][-5:])}\n"
            )

        # Collect all existing skill titles (for deduplication hint to the LLM)
        existing_titles = [s['title'] for s in current_skills.get('general_skills', [])]
        for task_type, skills in current_skills.get('task_specific_skills', {}).items():
            for s in skills:
                existing_titles.append(f"[{task_type}] {s.get('title', '')}")

        # Show the LLM what IDs to use (we'll reassign them anyway, but
        # providing the range avoids confusion in the returned JSON)
        example_ids = ", ".join(
            f'"dyn_{next_dyn_idx + j:03d}"'
            for j in range(self.max_new_skills_per_update)
        )

        return f"""Analyze these failed agent trajectories and suggest NEW skills to add to the skill bank.

FAILED TRAJECTORIES:
{''.join(failure_examples)}

EXISTING SKILL TITLES (avoid duplicating these):
{existing_titles}

Generate 1-{self.max_new_skills_per_update} NEW actionable skills that would help avoid these failures.
Each skill must have: skill_id, title (3-5 words), principle (1-2 sentences), when_to_apply.

Use skill_ids: {example_ids}

Return ONLY a JSON array of skills, no other text.
Example format:
[{{"skill_id": "dyn_{next_dyn_idx:03d}", "title": "Verify Object Location First", "principle": "Before attempting to pick up an object, always verify its current location by examining the environment.", "when_to_apply": "When the task requires moving an object but its location is uncertain"}}]
"""

    def _format_trajectory(self, steps: List[Dict],
                           action_limit: int = 150, obs_limit: int = 200) -> str:
        """Format trajectory steps for LLM analysis.

        Valid steps are shown individually with their observation and think snippet.
        Consecutive invalid steps are collapsed into a single summary line to reduce noise.
        Backfills 'valid' for old-format steps that lack the field (treats non-empty action as valid).
        """
        if not steps:
            return "  (no steps recorded)"

        lines = []
        step_num = 0
        invalid_run = 0  # consecutive invalid steps counter

        def _flush_invalid(count: int, start_num: int) -> str:
            if count == 1:
                return f"  Step {start_num}: [FORMAT ERROR] action tag missing or malformed — env returned nothing"
            return (f"  Steps {start_num}–{start_num + count - 1}: "
                    f"[FORMAT ERROR ×{count}] repeated malformed actions — env returned nothing each time")

        invalid_run_start = 0

        for step in steps:
            action = step.get('action', '').strip()
            obs    = step.get('observation', '').strip()
            # Backcompat: old format has no 'valid' key; infer from action content
            valid  = step.get('valid', 1 if action else 0)

            step_num += 1

            if not valid:
                if invalid_run == 0:
                    invalid_run_start = step_num
                invalid_run += 1
                continue

            # Flush any accumulated invalid run before this valid step
            if invalid_run > 0:
                lines.append(_flush_invalid(invalid_run, invalid_run_start))
                invalid_run = 0

            # Valid step — show full detail
            lines.append(f"  Step {step_num}:")
            lines.append(f"    Action: {action[:action_limit]}")
            lines.append(f"    Env:    {obs[:obs_limit]}")

        # Flush trailing invalid run
        if invalid_run > 0:
            lines.append(_flush_invalid(invalid_run, invalid_run_start))

        return '\n'.join(lines) if lines else "  (all steps had format errors)"

    @staticmethod
    def _strip_think_tags(text: str) -> str:
        """Remove <think>...</think> blocks emitted by reasoning models (e.g. MiniMax-M2.7)."""
        return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    def _parse_skills_response(self, response: str) -> List[Dict]:
        try:
            response = self._strip_think_tags(response)
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            if json_start != -1 and json_end > json_start:
                skills = json.loads(response[json_start:json_end])
                return [
                    s for s in skills
                    if all(k in s for k in ['skill_id', 'title', 'principle'])
                ]
        except json.JSONDecodeError as e:
            print(f"[SkillUpdater] JSON parse error: {e}")
        return []

    def get_update_summary(self) -> Dict:
        if not self.update_history:
            return {'total_updates': 0, 'total_skills_generated': 0}
        return {
            'total_updates': len(self.update_history),
            'total_skills_generated': sum(h['num_skills_generated'] for h in self.update_history),
            'all_skill_ids': [sid for h in self.update_history for sid in h['skill_ids']],
        }

    # ------------------------------------------------------------------ #
    # Schema-v2: meta-dimension-aware failure analysis                    #
    # ------------------------------------------------------------------ #

    META_DIMS = ('D-PLAN', 'D-TRACK', 'D-EXEC', 'D-GUARD')

    def analyze_failures_with_meta(
        self,
        failed_trajectories: List[Dict],
        current_skills: Dict,
        focus_dims: Optional[List[str]] = None,
        success_trajs: Optional[List[Dict]] = None,
    ) -> Dict:
        """Analyse failed trajectories and return meta-dimension update instructions.

        Args:
            failed_trajectories: Same format as ``analyze_failures``.
            current_skills: Current skill bank dict.
            focus_dims: If provided, restrict LLM to only update these dimensions.
                        None means all dimensions are allowed.

        Returns:
            Dict with keys:
              ``updates``    – list of ``{skill_id, dim, new_content}`` dicts.
              ``new_skills`` – list of new skill dicts to add.
        """
        if not failed_trajectories:
            return {'updates': [], 'new_skills': []}

        # Validate focus_dims against known dimensions
        if focus_dims:
            focus_dims = [d for d in focus_dims if d in self.META_DIMS]
        if not focus_dims:
            focus_dims = None

        next_dyn_idx = self._next_dyn_index(current_skills)
        prompt = self._build_meta_analysis_prompt(
            failed_trajectories, current_skills, next_dyn_idx,
            focus_dims=focus_dims, success_trajs=success_trajs or [],
        )
        sep = '=' * 60
        print(f"[SkillUpdater] LLM query: n_failed={len(failed_trajectories)}, "
              f"n_success={len(success_trajs or [])}, "
              f"focus_dims={focus_dims or 'auto'}, "
              f"prompt_chars={len(prompt)}, prompt_tokens≈{len(prompt)//4}")
        _prompt_oneline = prompt.replace('\n', '\\n')
        print(f"[SkillUpdater] PROMPT: {_prompt_oneline}")
        try:
            content = _llm_chat(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                mis_id=self._mis_id,
                max_tokens=32768,
            )
            print(f"[SkillUpdater] raw response:\n{content}\n{'=' * 60}")
            if not content:
                print(f"[SkillUpdater] Empty response content")
                return {'updates': [], 'new_skills': []}
            result = self._parse_meta_response(content, next_dyn_idx, focus_dims=focus_dims)
            self.update_history.append({
                'num_failures_analyzed': len(failed_trajectories),
                'num_skills_generated': len(result['new_skills']),
                'num_meta_updates': len(result['updates']),
                'skill_ids': [s.get('skill_id') for s in result['new_skills']],
                'focus_dims': focus_dims,
            })
            return result
        except Exception as e:
            print(f"[SkillUpdater] analyze_failures_with_meta error: {e}")
            return {'updates': [], 'new_skills': []}

    MAX_CANDIDATE_GENERAL = 2
    MAX_CANDIDATE_TASK_SPECIFIC = 2

    def _collect_used_skills(
        self,
        failed_trajectories: List[Dict],
        current_skills: Dict,
    ) -> List[Dict]:
        """Return skills used in failed trajectories, sorted by usage frequency.

        Selects top-2 general skills and top-2 task-specific skills separately
        by frequency, so both categories are represented in the prompt.
        """
        freq: Dict[str, int] = {}
        for traj in failed_trajectories:
            for sid in (traj.get('skill_ids_used') or []):
                freq[sid] = freq.get(sid, 0) + 1

        if not freq:
            return []

        general_skills: List[Dict] = current_skills.get('general_skills', [])
        task_specific_skills: List[Dict] = [
            s for skills in current_skills.get('task_specific_skills', {}).values()
            for s in skills
        ]
        general_ids = {s['skill_id'] for s in general_skills}
        task_specific_ids = {s['skill_id'] for s in task_specific_skills}
        skill_map = {s['skill_id']: s for s in general_skills + task_specific_skills}

        top_general = sorted(
            [sid for sid in freq if sid in general_ids],
            key=lambda sid: freq[sid], reverse=True
        )[:self.MAX_CANDIDATE_GENERAL]

        top_task_specific = sorted(
            [sid for sid in freq if sid in task_specific_ids],
            key=lambda sid: freq[sid], reverse=True
        )[:self.MAX_CANDIDATE_TASK_SPECIFIC]

        result = [skill_map[sid] for sid in top_general + top_task_specific if sid in skill_map]
        print(f"[SkillUpdater] candidate skills: "
              f"general(top-{self.MAX_CANDIDATE_GENERAL})={[(sid, freq[sid]) for sid in top_general]}, "
              f"task_specific(top-{self.MAX_CANDIDATE_TASK_SPECIFIC})={[(sid, freq[sid]) for sid in top_task_specific]}")
        return result

    def _build_meta_analysis_prompt(
        self,
        failed_trajectories: List[Dict],
        current_skills: Dict,
        next_dyn_idx: int,
        focus_dims: Optional[List[str]] = None,
        success_trajs: Optional[List[Dict]] = None,
    ) -> str:
        # ── Dimension descriptions ────────────────────────────────────────
        all_dim_desc = {
            'D-PLAN':  'Overall strategy decomposition and search/exploration heuristics',
            'D-TRACK': 'Continuous tracking of object states, positions, and task progress',
            'D-EXEC':  'Fine-grained action execution details and operation sequences',
            'D-GUARD': 'Explicit constraints against common failure modes and error patterns',
        }
        allowed_dims = focus_dims if focus_dims else list(self.META_DIMS)
        allowed_dims_str = '/'.join(allowed_dims)

        dim_desc_lines = []
        for dim, desc in all_dim_desc.items():
            marker = ' ← FOCUS' if focus_dims and dim in focus_dims else ''
            dim_desc_lines.append(f"  {dim}: {desc}{marker}")
        dim_desc_block = '\n'.join(dim_desc_lines)

        # ── Current skills (reference) ────────────────────────────────────
        used_skills = self._collect_used_skills(failed_trajectories, current_skills)
        skill_detail_lines = []
        for s in used_skills:
            meta = s.get('meta', {})
            lines = [f"[{s['skill_id']}] {s.get('title', '')}  (when: {s.get('when_to_apply', '')})"]
            if s.get('synthesized_text'):
                lines.append(f"  Overall: {s['synthesized_text'][:300]}")
            for dim in self.META_DIMS:
                marker = ' ← FOCUS' if focus_dims and dim in focus_dims else ''
                lines.append(f"  [{dim}]: {meta.get(dim, '(none)')}{marker}")
            skill_detail_lines.append('\n'.join(lines))

        skills_block = (
            '\n\n'.join(skill_detail_lines)
            if skill_detail_lines
            else "(No skills were used in these failures.)"
        )

        # ── Success reference trajectories ────────────────────────────────
        success_block = ""
        if success_trajs:
            success_parts = []
            for i, traj in enumerate(success_trajs[:2], 1):
                ep_len = len(traj.get('trajectory', []))
                success_parts.append(
                    f"\nSuccess Example {i}:\n"
                    f"Task: {traj['task']}\n"
                    f"Episode: {ep_len} steps (SUCCESS)\n"
                    f"Full trajectory:\n"
                    f"{self._format_trajectory(traj.get('trajectory', []))}\n"
                )
            success_block = "SUCCESS REFERENCE (same task type — what worked):\n" + ''.join(success_parts)

        # ── Failed trajectories ───────────────────────────────────────────
        failure_parts = []
        for i, traj in enumerate(failed_trajectories[:10], 1):
            ep_len = len(traj.get('trajectory', []))
            used = traj.get('skill_ids_used') or []
            failure_parts.append(
                f"\nFailed Example {i}:\n"
                f"Task: {traj['task']}\n"
                f"Task Type: {traj['task_type']}\n"
                f"Episode: {ep_len} steps (FAILED)\n"
                f"Skills used: {used}\n"
                f"Full trajectory:\n"
                f"{self._format_trajectory(traj.get('trajectory', []))}\n"
            )
        failures_block = "FAILED TRAJECTORIES:\n" + ''.join(failure_parts)

        # ── Focus / update instructions ───────────────────────────────────
        if focus_dims:
            focus_note = (
                f"FOCUS: For this batch, concentrate analysis on dimensions [{allowed_dims_str}] "
                f"— these are most likely to be the bottleneck at the current training stage. "
                f"You may still update other dimensions if the root cause clearly points there."
            )
        else:
            focus_note = "You may update any of the four dimensions based on your analysis."

        if used_skills:
            update_task = (
                f"For each FAILED trajectory, identify the skill and dimension that caused the failure, "
                f"then provide an improved value. {focus_note}"
            )
        else:
            update_task = (
                f"No skills were used. Propose new skills to address the failure patterns. "
                f"Do not put anything in 'updates'. {focus_note}"
            )

        example_new_id = f"dyn_{next_dyn_idx:03d}"

        return f"""You are improving an agent skill bank based on failed task trajectories.
You have access to SUCCESSFUL and FAILED trajectories for comparison.

SKILL DIMENSIONS:
{dim_desc_block}

CURRENT SKILLS (reference — these may need updating):
{skills_block}

{success_block}
{failures_block}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 1 — ROOT CAUSE ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
For EACH failed trajectory:
  a) Compare with the SUCCESS reference: what did the successful agent do differently?
  b) Identify the failure dimension:
     - D-PLAN  if the agent used wrong strategy, wrong search order, or wrong sub-goal sequencing
     - D-TRACK if the agent lost track of object location, state, or task progress counter
     - D-EXEC  if the agent performed wrong action sequence, missed a required step, or used wrong appliance
     - D-GUARD if the agent violated a known constraint (premature 'done', skipped transformation, etc.)
  c) Which skill (if any) contributed to this failure? Which specific dimension?

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 2 — UPDATE INSTRUCTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{update_task}

NEW SKILL CRITERIA — add a new skill ONLY if ALL conditions hold:
  1. The failure pattern appears across multiple trajectories in this batch
  2. No existing skill (including its D-GUARD) addresses this pattern, even partially
  3. The pattern is specific and actionable (not a vague restatement of existing guidance)
  4. You CANNOT improve an existing skill's D-GUARD to cover this pattern
  → When in doubt: update D-GUARD of the most relevant existing skill rather than adding new.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Return a JSON object with exactly two keys:
- "updates": list of objects, each with:
    "skill_id": id of the skill to update (must be one listed in CURRENT SKILLS above),
    "dim": one of {allowed_dims_str},
    "new_content": improved 1-3 sentence content for that dimension
- "new_skills": list of new skill objects (only when criteria above are met), each with:
    "skill_id": "{example_new_id}",
    "title": 3-5 words,
    "when_to_apply": one sentence,
    "meta": include only the dimensions relevant to this skill from [D-PLAN, D-TRACK, D-EXEC, D-GUARD]

Return ONLY the JSON object, no markdown, no preamble, no explanation.
Example: {{"updates": [{{"skill_id": "gen_001", "dim": "{allowed_dims[0]}", "new_content": "Improved content here."}}], "new_skills": []}}
"""

    def _parse_meta_response(
        self,
        response: str,
        next_dyn_idx: int,
        focus_dims: Optional[List[str]] = None,
    ) -> Dict:
        try:
            # Strip <think>...</think> blocks emitted by reasoning models
            response = self._strip_think_tags(response)
            # Strip markdown code fences if present (e.g. ```json ... ```)
            response = re.sub(r'^```[^\n]*\n', '', response.strip())
            response = re.sub(r'\n?```$', '', response.strip())
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                raw = response[json_start:json_end]
                raw = raw.replace('\r\n', ' ').replace('\r', ' ').replace('\n', ' ')
                data = json.loads(raw)
                allowed = set(focus_dims) if focus_dims else set(self.META_DIMS)
                updates = [
                    u for u in data.get('updates', [])
                    if all(k in u for k in ('skill_id', 'dim', 'new_content'))
                    and u['dim'] in self.META_DIMS
                    and u['dim'] in allowed
                ]
                if focus_dims:
                    all_updates = data.get('updates', [])
                    dropped = len(all_updates) - len(updates)
                    if dropped:
                        print(f"[SkillUpdater] Dropped {dropped} updates outside focus_dims={focus_dims}")
                raw_new = data.get('new_skills', [])
                # 修复: LLM 有时会把 meta 返回成 list of dict，统一转换为 dict
                for s in raw_new:
                    meta = s.get('meta')
                    if isinstance(meta, list):
                        merged = {}
                        for item in meta:
                            if isinstance(item, dict):
                                merged.update(item)
                        s['meta'] = merged
                new_skills = self._reassign_dyn_ids(
                    [s for s in raw_new if 'skill_id' in s and 'title' in s],
                    next_dyn_idx,
                )
                return {
                    'updates': updates,
                    'new_skills': new_skills[:self.max_new_skills_per_update],
                }
        except json.JSONDecodeError as e:
            print(f"[SkillUpdater] _parse_meta_response JSON error: {e}")
        return {'updates': [], 'new_skills': []}
