# ============================================================
# train_a2c_poker.py
# A2C training on unified Poker-v0 environment (PettingZoo Texas Hold'em)
# Assumes poker_env.py has already registered "Poker-v0"
# ============================================================

import os
import ray

from ray.tune.registry import register_env
from ray.rllib.algorithms.a2c import A2CConfig

# ⚠️ 根据你的路径修改这一行：
# 如果 poker_env.py 在项目根目录，就用下面这一行
from src.env.poker_env import env_creator

# 如果你是放在 src/env/poker_env.py，可以改成：
# from src.env.poker_env import env_creator


# ------------------------------------------------------------
# 1. 注册环境（如果在 poker_env.py 里已经 register 过，这里再写一遍也没关系）
# ------------------------------------------------------------
def _env_creator(config=None):
    return env_creator(config or {})

register_env("Poker-v0", _env_creator)


# ------------------------------------------------------------
# 2. 初始化 Ray
# ------------------------------------------------------------
ray.init(ignore_reinit_error=True)


# ------------------------------------------------------------
# 3. 配置 A2C：共享一个 policy 给两个玩家
# ------------------------------------------------------------
config = (
    A2CConfig()
    .environment(env="Poker-v0")
    .framework("torch")
    .rollouts(
        num_rollout_workers=2,      # 你电脑很卡的话可以改成 1
        rollout_fragment_length=50
    )
    .training(
        gamma=0.99,
        lr=7e-4,
        use_critic=True,
        use_gae=True,
        grad_clip=0.5,
    )
    .resources(
        num_gpus=0                  # 如果有 GPU 可以改成 1
    )
)

# Multi-agent 设置：所有 agent 都用同一个 policy（shared_policy）
config = config.multi_agent(
    policies={
        "shared_policy": (
            None,   # policy class (None -> default)
            None,   # obs space (None -> infer from env)
            None,   # act space (None -> infer from env)
            {},     # policy config
        )
    },
    policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
    policies_to_train=["shared_policy"],
)


# ------------------------------------------------------------
# 4. 构建 A2C 算法
# ------------------------------------------------------------
algo = config.build()

# 创建保存 checkpoint 的目录
checkpoint_dir = "checkpoints_a2c"
os.makedirs(checkpoint_dir, exist_ok=True)


# ------------------------------------------------------------
# 5. 训练循环
# ------------------------------------------------------------
num_iters = 80        # 可以先跑 20–50 看效果，再加
save_every = 10       # 每隔多少 iter 存一次模型

for i in range(num_iters):
    result = algo.train()
    mean_reward = result.get("episode_reward_mean", None)
    episodes = result.get("episodes_this_iter", None)

    print(f"Iter {i+1:03d} | "
          f"mean_reward={mean_reward:.3f} | "
          f"episodes={episodes}")

    # 每隔一段时间存个 checkpoint
    if (i + 1) % save_every == 0:
        ckpt_path = algo.save(checkpoint_dir)
        print(f"  💾 Saved checkpoint at: {ckpt_path}")

print("\n✅ A2C training finished.")

ray.shutdown()
