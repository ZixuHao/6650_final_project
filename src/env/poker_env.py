# ============================================================
# poker_env.py
# Unified Poker Environment for Benchmarking
# Compatible with:
# - PettingZoo texas_holdem_v4
# - RLlib MultiAgentEnv wrapper
# - Action masking
# - Shared policy / independent policy
# - Baseline AEC loops
# ============================================================

import numpy as np
from pettingzoo.classic import texas_holdem_v4
from ray.rllib.env import PettingZooEnv
from gymnasium.spaces import Dict, Box, Discrete
from ray.tune.registry import register_env


# ------------------------------------------------------------
# Optional Reward Wrapper (Optional but recommended)
# ------------------------------------------------------------

class RaisePenaltyWrapper(PettingZooEnv):
    """
    This wrapper slightly penalizes constant RAISE behavior and
    makes win/loss rewards asymmetric. Many of your groupmates
    already used something similar, so we add it as an option.
    """

    RAISE_ACTION = 1  # for reference: [fold=0, raise=1, call=2, check=3]

    def __init__(self, env, enable_penalty=False,
                 base_raise_penalty=0.05,
                 win_scale=0.75,
                 lose_scale=1.50):
        super().__init__(env)
        self.enable_penalty = enable_penalty
        self.base_raise_penalty = base_raise_penalty
        self.win_scale = win_scale
        self.lose_scale = lose_scale

    def step(self, action_dict):
        obs, rewards, terminations, truncations, infos = super().step(action_dict)

        # If penalty is disabled → return raw results
        if not self.enable_penalty:
            return obs, rewards, terminations, truncations, infos

        # Otherwise apply raise penalty & scaling
        new_rewards = {}
        for agent, r in rewards.items():
            a = action_dict.get(agent, None)

            # Penalize raise a little bit
            if a == self.RAISE_ACTION:
                r -= self.base_raise_penalty

            # Scale endgame rewards (optional)
            done = terminations.get(agent, False) or truncations.get(agent, False)
            if done:
                if r > 0:
                    r = self.win_scale * r
                elif r < 0:
                    r = self.lose_scale * r

            new_rewards[agent] = r

        return obs, new_rewards, terminations, truncations, infos


# ------------------------------------------------------------
# Unified Environment Creator
# ------------------------------------------------------------

def env_creator(config=None):
    """
    Creates a unified Poker environment. Works for:
    - RLlib (via register_env)
    - Stable-Baselines3 (by wrapping manually)
    - Baseline rule-based models (AECEnv)
    """

    raw_env = texas_holdem_v4.env()

    # seed (optional)
    if config is not None and "seed" in config:
        raw_env.reset(seed=config["seed"])

    # Wrap raw env → MultiAgentEnv for RLlib
    base_env = PettingZooEnv(raw_env)

    # You can choose penalty ON/OFF:
    # If your team doesn't want penalty, set enable_penalty=False
    wrapper_env = RaisePenaltyWrapper(
        base_env,
        enable_penalty=config.get("enable_penalty", False) if config else False
    )

    return wrapper_env


# ------------------------------------------------------------
# Register environment for RLlib
# ------------------------------------------------------------

register_env("Poker-v0", lambda config: env_creator(config))


# ------------------------------------------------------------
# Helper function to use env easily without RLlib
# ------------------------------------------------------------

def make_env(render=False, seed=None, penalty=False):
    """
    For SB3 or manual eval:
        env = make_env()
    """
    raw_env = texas_holdem_v4.env(
        render_mode="human" if render else None
    )

    if seed is not None:
        raw_env.reset(seed=seed)

    # Convert AEC → MultiAgent so that obs/action spaces match RLlib defaults
    base_env = PettingZooEnv(raw_env)

    return RaisePenaltyWrapper(base_env, enable_penalty=penalty)
