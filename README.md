# 6650 Final Project — Texas Hold’em Reinforcement Learning Benchmark

This project implements, evaluates, and compares multiple RL agents in the **PettingZoo Texas Hold’em** environment:

**Rule-based policy**
**Shallow Q-learning**
**A2C baseline (our work)**
PPO baseline (group member)
DQN baseline (group member)

The goal is to build a **reproducible benchmark suite** and analyze performance across classical and deep RL methods.

---

# Directory Structure

```
Baseline Policies/
    rule_based.py
    shallow_q.py
    baseline_analysis.ipynb
DeepRL/
    A2C/
        train_a2c.ipynb
        single_agent_wrapper.py
        generate_poker_gif.py
PPO/
DQN/
runs/
    a2c_run_1/
    a2c_run_2/
    ...
README.md
```

---

# Rule-based Baseline & Shallow Q-Learning

This module implements and evaluates two essential baseline agents for the Texas Hold’em environment (`pettingzoo.classic.texas_holdem_v4`):

1. **Rule-based Policy (hand-crafted expert system)**
2. **Shallow RL Agent (Tabular Q-learning)**

These baselines provide reference performance before introducing Deep RL (PPO / DQN / A2C).

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

# Rule-based Action Distribution

*(Plot saved as `plots/rule_based_action_dist.png`)*
![Rule-based Action Distribution](plots/rule_based_action_dist.png)

---

# Rule-based Performance vs Random

| Agent      | Win Rate | Tie Rate | Loss Rate | Mean Reward |
| ---------- | -------- | -------- | --------- | ----------- |
| Rule-based | 0.600    | 0.012    | 0.388     | 0.568       |

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

# Learning Curve

![Learning Curve](plots/q_learning_curve.png)

---

# Q-table Heatmap

![Q-table Heatmap](plots/q_table_heatmap.png)

---

# Shallow Q-learning Performance vs Random

| Agent              | Win Rate | Tie Rate | Loss Rate | Mean Reward |
| ------------------ | -------- | -------- | --------- | ----------- |
| Shallow Q-learning | 0.954    | 0.004    | 0.042     | 1.990       |

---

# Rule-based vs Shallow Q-learning

| Agent              | Mean Reward | Win Rate |
| ------------------ | ----------- | -------- |
| Rule-based         | 0.568       | 0.600    |
| Shallow Q-learning | 1.990       | 0.954    |

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
✔ Ready for final benchmark comparison (A2C vs PPO vs DQN)
