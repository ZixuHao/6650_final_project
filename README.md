# 6650 Final Project — Texas Hold’em Reinforcement Learning Benchmark

This project implements, evaluates, and compares multiple RL agents in the **PettingZoo Texas Hold’em** and **RLCard**environment:

**Rule-based policy**
**Shallow Q-learning**
**A2C baseline**
**PPO**


The goal is to build a **reproducible benchmark suite** and analyze performance across classical and deep RL methods.

---

# Directory Structure

```
.
|-- A2C
|   |-- a2c_poker_model.zip
|   |-- a2c_vs_rule_head2head.png
|   |-- rule_vs_a2c.png
|   |-- runs
|   |   |-- a2c_run_1
|   |   |   |-- a2c_model.zip
|   |   |   |-- after_training.gif
|   |   |   |-- before_training.gif
|   |   |   |-- reward_history.pkl
|   |   |   `-- training_curve.png
|   |   |-- a2c_run_2
|   |   |   |-- a2c_model.zip
|   |   |   |-- before_training_verbose.gif
|   |   |   |-- reward_history.pkl
|   |   |   `-- training_curve.png
|   |   `-- a2c_run_3
|   |       |-- a2c_model.zip
|   |       |-- after_training_verbose.gif
|   |       |-- before_training_verbose.gif
|   |       |-- reward_history.pkl
|   |       `-- training_curve.png
|   `-- train_a2c_poker.ipynb
|-- Baseline Policies
|   |-- RuleBased_Shallow.ipynb
|   |-- table_head_to_head.png
|   `-- table_performance_random.png
|-- README.md
|-- checkpoints_2
|   |-- algorithm_state.pkl
|   |-- policies
|   |   |-- bot
|   |   |   |-- policy_state.pkl
|   |   |   `-- rllib_checkpoint.json
|   |   `-- learner
|   |       |-- policy_state.pkl
|   |       `-- rllib_checkpoint.json
|   `-- rllib_checkpoint.json
|-- checkpoints_3
|   |-- algorithm_state.pkl
|   |-- policies
|   |   |-- player_0
|   |   |   |-- policy_state.pkl
|   |   |   `-- rllib_checkpoint.json
|   |   `-- player_1
|   |       |-- policy_state.pkl
|   |       `-- rllib_checkpoint.json
|   `-- rllib_checkpoint.json
|-- checkpoints_3_sep
|   |-- algorithm_state.pkl
|   |-- policies
|   |   |-- player_0
|   |   |   |-- policy_state.pkl
|   |   |   `-- rllib_checkpoint.json
|   |   `-- player_1
|   |       |-- policy_state.pkl
|   |       `-- rllib_checkpoint.json
|   `-- rllib_checkpoint.json
|-- checkpoints_3_sep_30epoch
|   |-- algorithm_state.pkl
|   |-- policies
|   |   |-- player_0
|   |   |   |-- policy_state.pkl
|   |   |   `-- rllib_checkpoint.json
|   |   `-- player_1
|   |       |-- policy_state.pkl
|   |       `-- rllib_checkpoint.json
|   `-- rllib_checkpoint.json
|-- checkpoints_3_share_10epoch
|   |-- algorithm_state.pkl
|   |-- policies
|   |   `-- shared_policy
|   |       |-- policy_state.pkl
|   |       `-- rllib_checkpoint.json
|   `-- rllib_checkpoint.json
|-- checkpoints_3_shared
|   |-- algorithm_state.pkl
|   |-- policies
|   |   `-- shared_policy
|   |       |-- policy_state.pkl
|   |       `-- rllib_checkpoint.json
|   `-- rllib_checkpoint.json
|-- ppo
|   |-- PPO_adversarial
|   |   |-- train_poke.ipynb
|   |   |-- train_poke_v2.ipynb
|   |   |-- train_poke_v2_DQN.ipynb
|   |   |-- train_poke_v2_seperate.ipynb
|   |   |-- train_poke_v2_seperate_show.ipynb
|   |   `-- train_poke_v2_shared.ipynb
|   |-- baseline_comparison.png
|   |-- checkpoints_2
|   |   |-- algorithm_state.pkl
|   |   |-- policies
|   |   |   |-- bot
|   |   |   |   |-- policy_state.pkl
|   |   |   |   `-- rllib_checkpoint.json
|   |   |   `-- learner
|   |   |       |-- policy_state.pkl
|   |   |       `-- rllib_checkpoint.json
|   |   `-- rllib_checkpoint.json
|   |-- ppo_poke.gif
|   |-- table_ppo_vs_random.png
|   `-- train_poke_PPO_unit_strategy.ipynb
|-- ppo_poke_sep.gif
|-- pyproject.toml
`-- share_vs_sep_policy_performance
    |-- table_2aa.png
    `-- table_spa.png
    
```

---

# Rule-based Baseline & Shallow Q-Learning

This module implements and evaluates two essential baseline agents for the Texas Hold’em environment (`pettingzoo.classic.texas_holdem_v4`):

1. **Rule-based Policy (hand-crafted expert system)**
2. **Shallow RL Agent (Tabular Q-learning)**

These baselines provide reference performance before introducing Deep RL (PPO / A2C).

---

# Environment Setup

```python
from pettingzoo.classic import texas_holdem_v4

```

Each agent receives:

* A 52-bit card visibility vector
* A legal action mask
* Actions: `0=Call`, `1=Raise`, `2=Fold`, `3=Check`

---

# 1. Rule-based Policy (Baseline)

## State Representation

We extract:

* **Street**: preflop, flop, turn, river → 4 levels
* **Hand strength**: weak, medium, strong → 3 levels

## Policy Logic

| Street          | Strength | Action Priority      |
| --------------- | -------- | -------------------- |
| Preflop         | Strong   | Raise → Call → Check |
| Preflop         | Medium   | Call → Check         |
| Preflop         | Weak     | Check → Fold         |
| Flop/Turn/River | Strong   | Raise → Call → Check |
| Flop/Turn/River | Medium   | Call → Check         |
| Flop/Turn/River | Weak     | Check → Fold         |

---


# Rule-based Performance vs Random

| Agent      | Win Rate | Tie Rate | Loss Rate | Mean Reward |
| ---------- | -------- | -------- | --------- | ----------- |
| Rule-based | 0.582    | 0.002    | 0.416     | 0.460       |

---

# 2. Shallow RL Agent — Tabular Q-learning

## State Encoding

12 discrete states formed by:

* 4 street levels
* 3 strength levels

## Q-learning Settings

| Parameter       | Value      |
| --------------- | ---------- |
| Episodes        | 5000       |
| Learning rate α | 0.1        |
| Discount γ      | 0.9        |
| ε-greedy        | 0.2 → 0.05 |
| Opponent        | Random     |

---


# Shallow Q-learning Performance vs Random

| Agent              | Win Rate | Tie Rate | Loss Rate | Mean Reward |
| ------------------ | -------- | -------- | --------- | ----------- |
| Shallow Q-learning | 0.928    | 0.002    | 0.07     | 2.002       |

---

# Rule-based vs Shallow Q-learning

| Agent              | Mean Reward | Win Rate |
| ------------------ | ----------- | -------- |
| Rule-based         | 0.460       | 0.582    |
| Shallow Q-learning | 2.002       | 0.928    |

---

# Completed Work (Baselines)

✔ Rule-based policy
✔ Rule-based action distribution
✔ Rule-based evaluation
✔ Shallow Q-learning
✔ Q-learning learning curve
✔ Q-table visualization
✔ Shallow Q-learning performance
✔ Rule-based vs Shallow comparison

---

# **3. Deep RL Baseline — A2C (Advantage Actor–Critic)**

This module provides a deep RL baseline using **Stable-Baselines3 A2C**.
A2C bridges the gap between our classical baselines and the more advanced PPO/DQN models.

---

# Environment Wrapper (SB3-compatible)

Texas Hold’em is multi-agent; SB3 is single-agent.
We implement a custom wrapper that:

* Flattens observations into SB3-friendly Box spaces
* Applies **illegal action masking**
* Uses a random opponent as baseline
* Supports `"rgb_array"` rendering for GIF generation

The wrapper enables stable SB3 training without crashes.

---

# A2C Training Configuration

| Parameter      | Value     |
| -------------- | --------- |
| Algorithm      | A2C       |
| Timesteps      | 200,000   |
| Learning rate  | 7e-4      |
| Gamma          | 0.99      |
| Opponent       | Random    |
| Policy Network | MLP       |
| Renderer       | rgb_array |

Training script:
`DeepRL/A2C/train_a2c.ipynb`

---

# A2C Training Curve (Reward vs Random)

*(Saved as `runs/a2c_run_X/training_curve.png`)*

![A2C Curve](runs/a2c_run_X/training_curve.png)

A2C shows mild improvement early on but stabilizes near the performance of a random baseline — expected for poker due to partial observability, stochasticity, and multi-agent dynamics.

---

# Visualization — Before vs After Training (GIF)

We generate poker replays with:

* Player identity
* Visible hole cards + community cards
* Chosen actions (Fold / Call / Raise / Check / Bet)
* Reward progression
* Full poker table GUI rendering

### Before Training (Random Agent)

```
runs/a2c_run_X/before_training_verbose.gif
```

### After Training (A2C Agent)

```
runs/a2c_run_X/after_training_verbose.gif
```

These GIFs match the format used by PPO/DQN teammates for unified benchmarking.

---

# A2C Performance vs Random

| Agent      | Win Rate   | Mean Reward | Notes                             |
| ---------- | ---------- | ----------- | --------------------------------- |
| A2C (ours) | ~0.48–0.52 | Around 0    | Slight improvement, high variance |
| Random     | 0.50       | 0           | baseline                          |

**Interpretation:**
A2C learns slightly more consistent betting behavior but remains close to random — highlighting that basic deep RL struggles on poker without opponent modeling or state abstraction.

---

# Baseline Comparison Summary

| Agent              | Strength                       | Weakness                            |
| ------------------ | ------------------------------ | ----------------------------------- |
| Rule-based         | Solid heuristics               | Not adaptive                        |
| Shallow Q-learning | Very strong vs random          | Limited state space                 |
| **A2C (ours)**     | Deep RL function approximation | Near-random due to poker complexity |

---

# A2C File Structure

```
DeepRL/A2C/
    train_a2c.ipynb
    single_agent_wrapper.py
    generate_poker_gif.py
runs/a2c_run_X/
    a2c_model.zip
    reward_history.pkl
    before_training_verbose.gif
    after_training_verbose.gif
    training_curve.png
```

---

# How to Run A2C

Open:

```
DeepRL/A2C/train_a2c.ipynb
```

Running all cells will generate:

* Saved SB3 A2C model
* Reward curve
* High-quality before/after training GIFs
* Evaluation logs

---

# Project Summary (Updated)

✔ Rule-based baseline
✔ Shallow Q-learning baseline
✔ A2C deep RL baseline (**this work**)
✔ Complete evaluation suite
✔ Unified visualization (GIFs, curves, tables)
✔ Ready for final benchmark comparison (Rule-based baseline vs Shallow Q Learning vs A2C vs PPO)
