import gymnasium as gym
import flappy_bird_gymnasium  # 必须import一下才能注册环境
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

# 1. 创建环境（不用显示，先专心训练）
def make_env():
    env = gym.make(
        "FlappyBird-v0",
        render_mode=None,     # 训练时不渲染，加速
        use_lidar=False       # 关闭 lidar，用较简单的低维特征
    )
    env = Monitor(env)        # 方便记录奖励
    return env

env = make_env()

# 2. 配置日志（可选，但有的话更好看训练过程）
log_dir = "./flappy_logs/"
new_logger = configure(log_dir, ["stdout", "csv"])

# 3. 建一个最基础的 DQN 模型
model = DQN(
    policy="MlpPolicy",
    env=env,
    learning_rate=1e-4,
    buffer_size=50_000,
    learning_starts=1_000,
    batch_size=32,
    gamma=0.99,
    train_freq=4,
    target_update_interval=1_000,
    verbose=1,
)

model.set_logger(new_logger)

# 4. 开始训练
# 步数可以先少一点，比如 100_000，跑得动再往上加
model.learn(total_timesteps=1000_000)

# 5. 保存模型
model.save("flappy_dqn_simple")

env.close()
print("Training done and model saved as flappy_dqn_simple.zip")
