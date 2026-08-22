#!/bin/bash
# WebShop + GRPO + Skill Evolution
# 用法: bash run_webshop_gigpo.sh [ENGINE] [GPU_NUM]
set -x
export HYDRA_FULL_ERROR=1
ulimit -n 65535
ulimit -u unlimited
ulimit -s unlimited
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export VLLM_ATTENTION_BACKEND=XFORMERS

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_FILE="${REPO_ROOT}/config.sh"

if [ ! -f "${CONFIG_FILE}" ]; then
    echo "错误: 未找到 ${CONFIG_FILE}"
    echo "请先执行: cp ${REPO_ROOT}/config.sh.example ${CONFIG_FILE} 并填写你的配置"
    exit 1
fi
source "${CONFIG_FILE}"

SKILLRL=${SKILLRL_ROOT}
DATA_DIR=${SKILLRL_VERL_DATA}
MODEL_PATH=${SKILLRL_MODEL_PATH}

# WebShop conda 环境（含 gym/webshop/java 依赖）
ENV_WEBSHOP=/path/to/workspace/env/webshop-cty
PYTHON=${ENV_WEBSHOP}/bin/python3

ENGINE=${1:-vllm}
GPU_NUM=${2:-8}

# ============================================================
# Step 1: 准备数据
# ============================================================
if [ ! -f "$DATA_DIR/train.parquet" ]; then
    echo "=== 准备 verl-agent parquet 数据 ==="
    mkdir -p $DATA_DIR
    cd $SKILLRL
    ${PYTHON} -m examples.data_preprocess.prepare \
        --mode 'text' \
        --train_data_size 16 \
        --val_data_size 128
fi

# ============================================================
# Step 2: 设置环境变量
# ============================================================
export PYTHONPATH=${SKILLRL}:${SKILLRL}/agent_system/environments/env_package/webshop/webshop:${PYTHONPATH}
export PYTHONNOUSERSITE=1
export JAVA_HOME=${ENV_WEBSHOP}/lib/jvm
export PATH=${JAVA_HOME}/bin:${PATH}

export RAY_memory_monitor_refresh_ms=0
export RAY_memory_usage_threshold=1.0
export RAY_DISABLE_MEMORY_MONITOR=1
export RAY_DISABLE_DASHBOARD=1
export SKILLRL_LLM_BACKEND=${SKILLRL_LLM_BACKEND}
export SKILLRL_LLM_MODEL=${SKILLRL_LLM_MODEL}

cd $SKILLRL

# ============================================================
# Step 3: 路径和时间戳
# ============================================================
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MODEL_NAME=$(basename $MODEL_PATH)
CKPT_DIR=${SKILLRL_EXP_BASE}/checkpoints/webshop/webshop-${MODEL_NAME}-grpo-skillevol-${TIMESTAMP}
TB_DIR=${SKILLRL_EXP_BASE}/tensorboard/webshop-${MODEL_NAME}-grpo-skillevol-${TIMESTAMP}
mkdir -p $CKPT_DIR $TB_DIR
export TENSORBOARD_DIR=$TB_DIR

# ============================================================
# Step 4: GPU 保活
# ============================================================
GPU_KEEPER_PID=""
if [ -z "${SKILLRL_GPU_KEEPER_SCRIPT}" ] || [ ! -f "${SKILLRL_GPU_KEEPER_SCRIPT}" ]; then
    echo "警告: GPU keeper 脚本不存在: ${SKILLRL_GPU_KEEPER_SCRIPT}"
else
    python3 ${SKILLRL_GPU_KEEPER_SCRIPT} --gpu_number ${GPU_NUM} --duration 0 \
        > /tmp/gpu_keeper_${TIMESTAMP}.log 2>&1 &
    GPU_KEEPER_PID=$!
    sleep 2
    echo "=== GPU keeper 已启动 (PID=${GPU_KEEPER_PID}) ==="
fi

_cleanup_gpu_keeper() {
    if [ -n "${GPU_KEEPER_PID}" ]; then
        kill ${GPU_KEEPER_PID} 2>/dev/null || true
    fi
}
trap _cleanup_gpu_keeper EXIT INT TERM

# ============================================================
# Step 5: 启动训练
# ============================================================
${PYTHON} -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=16 \
    data.val_batch_size=128 \
    data.max_prompt_length=6144 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    actor_rollout_ref.actor.checkpoint.contents="['model','hf_model','optimizer','extra']" \
    algorithm.use_kl_in_reward=False \
    env.env_name=Webshop \
    env.seed=0 \
    env.max_steps=15 \
    env.rollout.n=8 \
    env.resources_per_worker.num_cpus=0.1 \
    +env.use_skills_only_memory=True \
    +env.skills_only_memory.skills_json_path=memory_data/webshop/claude_style_skills.json \
    +env.skills_only_memory.retrieval_mode=embedding \
    +env.skills_only_memory.top_k=6 \
    +env.skills_only_memory.task_specific_top_k=5 \
    +env.skills_only_memory.ab_rollout=True \
    +env.skills_only_memory.ab_ratio=0.5 \
    +skill.ab_rollout=True \
    +skill.update_lower_bound=0.4 \
    +skill.prune_threshold=0.6 \
    +skill.update_freq=10 \
    +skill.max_new_skills=3 \
    +skill.min_utility=0.2 \
    +skill.merge_threshold=0.85 \
    +skill.val_episodes=16 \
    +skill.verify_method=rollout \
    +skill.focus_dims_mode=fixed \
    +skill.llm_model=${SKILLRL_LLM_MODEL} \
    +skill.save_path=memory_data/webshop/claude_style_skills_evolved_webshop-${MODEL_NAME}-gigpo-skillevol_${TIMESTAMP}.json \
    trainer.critic_warmup=0 \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name='skillrl_webshop' \
    trainer.experiment_name="webshop-${MODEL_NAME}-grpo-skillevol" \
    trainer.default_local_dir=${CKPT_DIR} \
    trainer.n_gpus_per_node=${GPU_NUM} \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=150 \
    trainer.val_before_train=True

echo "=== 停止 GPU 持续计算任务 ==="
[ -n "${GPU_KEEPER_PID}" ] && kill $GPU_KEEPER_PID 2>/dev/null || true
echo "=== 训练完成 ==="
