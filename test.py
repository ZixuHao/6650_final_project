import gymnasium as gym
import flappy_bird_gymnasium
from stable_baselines3 import DQN
import time

# 1. 创建带渲染的环境
env = gym.make(
    "FlappyBird-v0",
    render_mode="human",
    use_lidar=False    # 和训练时保持一致
)

# 2. 加载训练好的模型
model = DQN.load("flappy_dqn_simple", env=env)

obs, _ = env.reset()
while True:
    # 用训练好的策略选择动作
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, _ = env.reset()
        # 防止窗口刷太快
        time.sleep(0.5)

env.close()
