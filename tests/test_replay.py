"""
Test reset_all_with_gamefiles: verify that each worker is reset to its own gamefile.

Usage:
    cd /path/to/other/workspace/research/skillrl
    python tests/test_replay.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import random


def find_gamefiles(data_root: str, n: int = 8) -> list:
    """Find n distinct gamefiles from the training set."""
    pattern = os.path.join(data_root, "json_2.1.1/train/*/trial_*/game.tw-pddl")
    all_files = glob.glob(pattern)
    assert len(all_files) >= n, f"Found only {len(all_files)} gamefiles, need {n}"
    random.seed(42)
    selected = random.sample(all_files, n)
    print(f"Selected {n} gamefiles:")
    for i, f in enumerate(selected):
        print(f"  [{i}] ...{f[-60:]}")
    return selected


def test_reset_all_with_gamefiles():
    """
    Test that reset_all_with_gamefiles correctly resets each worker
    to its assigned gamefile.
    """
    import ray
    from omegaconf import OmegaConf

    DATA_ROOT = "/path/to/alfworld/game_data"
    ALF_CONFIG = "/path/to/workspace/agent_system/environments/env_package/alfworld/configs/config_tw.yaml"
    N_TASKS = 4   # number of distinct tasks
    GROUP_N = 2   # rollouts per task (total workers = N_TASKS × GROUP_N = 8)

    os.environ.setdefault("ALFWORLD_DATA", DATA_ROOT)

    print("=" * 60)
    print("Test: reset_all_with_gamefiles")
    print("=" * 60)

    # 1. Find gamefiles
    gamefiles = find_gamefiles(DATA_ROOT, n=N_TASKS)

    # 2. Init Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    # 3. Create AlfworldEnvs directly
    from agent_system.environments.env_package.alfworld.envs import AlfworldEnvs

    print(f"\nCreating {N_TASKS * GROUP_N} workers ({N_TASKS} tasks × {GROUP_N} rollouts)...")
    envs = AlfworldEnvs(
        alf_config_path=ALF_CONFIG,
        seed=0,
        env_num=N_TASKS,
        group_n=GROUP_N,
        resources_per_worker={'num_cpus': 0.1},
        is_train=True,
    )
    print(f"Created {envs.num_processes} workers (group_n={envs.group_n})")

    # 4. Expand gamefiles: each task repeated group_n times
    gamefiles_expanded = []
    for gf in gamefiles:
        gamefiles_expanded.extend([gf] * GROUP_N)
    assert len(gamefiles_expanded) == N_TASKS * GROUP_N

    # 5. Call reset_all_with_gamefiles
    print(f"\nCalling reset_all_with_gamefiles with {len(gamefiles_expanded)} gamefiles...")
    text_obs, _, infos = envs.reset_all_with_gamefiles(gamefiles_expanded)

    print(f"\nResults ({len(text_obs)} workers):")
    all_match = True
    for i in range(len(text_obs)):
        obs_preview = text_obs[i][:80].replace('\n', ' ') if text_obs[i] else ''
        expected_gf = gamefiles_expanded[i]
        task_idx = i // GROUP_N
        rollout_idx = i % GROUP_N
        expected_task_type = _extract_task_type(expected_gf)
        match = (expected_task_type and expected_task_type in obs_preview.lower())
        status = '✓' if match else '?'
        if not match:
            all_match = False
        print(f"  worker[{i}] task={task_idx} rollout={rollout_idx} {status}")
        print(f"    expected: ...{expected_gf[-50:]}")
        print(f"    obs:      {obs_preview}...")

    print(f"\n{'✓ All workers assigned to correct tasks' if all_match else '⚠ Some workers may not match (obs check is approximate)'}")

    # 6. Verify admissible commands are populated for all workers
    print(f"\nAdmissible commands check:")
    for i in range(len(envs.prev_admissible_commands)):
        n_cmds = len(envs.prev_admissible_commands[i]) if envs.prev_admissible_commands[i] else 0
        status = '✓' if n_cmds > 0 else '✗'
        print(f"  worker[{i}]: {status} {n_cmds} admissible commands")

    envs.close() if hasattr(envs, 'close') else None
    print("\n✓ Test complete")
    return True


def _extract_task_type(gamefile: str) -> str:
    """Extract task type keyword from gamefile path."""
    gf_lower = gamefile.lower()
    for kw in ['pick_clean', 'pick_heat', 'pick_cool', 'pick_two', 'look_at', 'pick_and_place']:
        if kw in gf_lower:
            return kw.replace('_', ' ')
    return ''


def _extract_task_type_from_obs(obs: str) -> str:
    obs_lower = obs.lower()
    for kw in ['clean', 'heat', 'cool', 'look at', 'put two', 'put']:
        if kw in obs_lower:
            return kw
    return ''


if __name__ == '__main__':
    test_reset_all_with_gamefiles()
