# A2SE:

A2SE is a reinforcement learning framework for LLM agents that features an **Ability-Aligned Skill Evolution** mechanism. During training, the agent automatically summarizes, updates, and prunes a skill library via a strong LLM, and validates skill quality through A/B rollout experiments.

Built on [veRL](https://github.com/volcengine/verl).

---

## Installation

```bash
pip install -r requirements.txt
pip install vllm==0.11.0
pip install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
pip install -e .
```

---

## Configuration

All user-specific paths are managed via `config.sh`. **Create it before first run:**

```bash
cp config.sh.example config.sh
```

Then edit `config.sh` and fill in the following variables:

| Variable                    | Description                                                                 |
| -----------------------------| -----------------------------------------------------------------------------|
| `SKILLRL_USER_BASE`         | Your home directory on the cluster                                          |
| `SKILLRL_ROOT`              | Absolute path to this repository                                            |
| `SKILLRL_MODEL_PATH`        | Path to the pretrained base model (e.g., `Qwen2.5-7B-Instruct`)             |
| `SKILLRL_ALFWORLD_DATA`     | ALFWorld game data directory (must contain `json_2.1.1/train/`)             |
| `SKILLRL_VERL_DATA`         | Output directory for verl-agent parquet files (auto-generated on first run) |
| `SKILLRL_EXP_BASE`          | Root directory for experiment outputs (checkpoints / TensorBoard)           |
| `SKILLRL_LLM_BACKEND`       | LLM backend for skill evolution calls: `catpaw` or `aigc`                   |
| `SKILLRL_LLM_MODEL`         | LLM model name                                                              |
| `SKILLRL_GPU_KEEPER_SCRIPT` | *(Optional)* Path to GPU keep-alive script; leave empty to skip             |

> `config.sh` is gitignored and will never be committed.

---

## Data Preparation

### ALFWorld

Download the ALFWorld game files. After extraction, the directory structure should be:

```
$SKILLRL_ALFWORLD_DATA/
└── json_2.1.1/
    ├── train/        # 614 game directories
    └── valid_seen/
```

The training script automatically calls `examples/data_preprocess/prepare.py` on first run to generate verl-agent parquet files in `$SKILLRL_VERL_DATA`.

### WebShop

WebShop requires a dedicated conda environment with Java and gym-webshop dependencies:

```bash
cd agent_system/environments/env_package/webshop
./setup.sh -d all
```

After setup, edit `examples/skill_evolution_trainer/run_webshop_grpo.sh` and set `ENV_WEBSHOP` to your conda environment path:

```bash
# In run_webshop_grpo.sh, line ~29:
ENV_WEBSHOP=/path/to/your/webshop-conda-env
```

---

## Initial Skill Library

Pre-built skill libraries for both environments are provided in `memory_data/`:

```
memory_data/
├── alfworld/claude_style_skills.json   # Initial skill bank for ALFWorld
└── webshop/claude_style_skills.json    # Initial skill bank for WebShop
```

Evolved skill libraries are saved alongside these files with a timestamp suffix after training.

---

## Training

### ALFWorld — GRPO + Skill Evolution

```bash
bash examples/skill_evolution_trainer/run_alfworld_grpo.sh [ENGINE] [GPU_NUM]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `ENGINE` | `vllm` | Rollout engine |
| `GPU_NUM` | `8` | Number of GPUs |

**Example:**

```bash
bash examples/skill_evolution_trainer/run_alfworld_grpo.sh vllm 8
```

**Key hyperparameters (script defaults):**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `data.train_batch_size` | 16 | Training batch size per step |
| `env.max_steps` | 50 | Max steps per episode |
| `env.rollout.n` | 8 | Rollout samples per prompt |
| `skill.top_k` | 3 | Skills retrieved per episode |
| `skill.ab_ratio` | 0.5 | Fraction of rollouts that use skill memory (A/B split) |
| `skill.update_freq` | 10 | Epochs between skill evolution cycles |
| `skill.verify_method` | `rollout` | Skill validation method (`rollout` / `notverify`) |
| `trainer.total_epochs` | 150 | Total training epochs |

---

### WebShop — GRPO + Skill Evolution

```bash
bash examples/skill_evolution_trainer/run_webshop_grpo.sh [ENGINE] [GPU_NUM]
```

| Argument  | Default | Description    |
| -----------| ---------| ----------------|
| `ENGINE`  | `vllm`  | Rollout engine |
| `GPU_NUM` | `8`     | Number of GPUs |

**Example:**

```bash
bash examples/skill_evolution_trainer/run_webshop_grpo.sh vllm 8
```

> The script uses the Python interpreter from `$ENV_WEBSHOP/bin/python3` to satisfy WebShop's dependencies.


## Outputs

Results are organized under `$SKILLRL_EXP_BASE` by task and timestamp:

```
$SKILLRL_EXP_BASE/
├── checkpoints/
│   ├── alfworld/alfworld-{MODEL}-grpo-skillevol-{TIMESTAMP}/
│   └── webshop/webshop-{MODEL}-grpo-skillevol-{TIMESTAMP}/
└── tensorboard/
    ├── alfworld-{MODEL}-grpo-skillevol-{TIMESTAMP}/
    └── webshop-{MODEL}-grpo-skillevol-{TIMESTAMP}/
```

Monitor training:

```bash
tensorboard --logdir $SKILLRL_EXP_BASE/tensorboard
```

---
