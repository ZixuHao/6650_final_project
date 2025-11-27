# 6650_final_project


# 🃏 Rule-based Baseline & Shallow Q-Learning

This module implements and evaluates two essential baseline agents for the Texas Hold’em environment (`pettingzoo.classic.texas_holdem_v4`):

1. **Rule-based Policy (hand-crafted expert system)**
2. **Shallow RL Agent (Tabular Q-learning)**

These baselines provide reference performance before introducing Deep RL (PPO).

---

# 📦 Environment Setup

```python
from pettingzoo.classic import texas_holdem_v4
```

Each agent receives:

* A 52-bit card visibility vector
* A legal action mask
* Actions: `0=Call`, `1=Raise`, `2=Fold`, `3=Check`

---

# 🎯 1. Rule-based Policy (Baseline)

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

# 📊 Rule-based Action Distribution

![Rule-based Action Distribution](rule_based_action_dist.png)

---

# 🏆 Rule-based Performance vs Random

| Agent      | Win Rate | Tie Rate | Loss Rate | Mean Reward |
| ---------- | -------- | -------- | --------- | ----------- |
| Rule-based | 0.600    | 0.012    | 0.388     | 0.568       |

---

# 🤖 2. Shallow RL Agent — Tabular Q-learning

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

# 📈 Learning Curve

![Learning Curve](q_learning_curve.png)

---

# 🔥 Q-table Heatmap

![Q-table Heatmap](q_table_heatmap.png)

---

# 🏅 Shallow Q-learning Performance vs Random

| Agent              | Win Rate | Tie Rate | Loss Rate | Mean Reward |
| ------------------ | -------- | -------- | --------- | ----------- |
| Shallow Q-learning | 0.954    | 0.004    | 0.042     | 1.990       |

---

# ⚔️ Rule-based vs Shallow Q-learning (Head-to-Head)

| Agent              | Mean Reward | Win Rate |
| ------------------ | ----------- | -------- |
| Rule-based         | 0.568       | 0.600    |
| Shallow Q-learning | 1.990       | 0.954    |

---

# 📚 Summary of Completed Work

✔ Rule-based policy
✔ Rule-based action distribution plot
✔ Rule-based performance table
✔ Shallow Q-learning implementation
✔ Q-learning learning curve
✔ Q-table heatmap
✔ Shallow Q-learning performance table
✔ Rule-based vs shallow Q-learning comparison table

This baseline module is fully completed.

---

# 📁 File Structure

| File                      | Description       |
| ------------------------- | ----------------- |
| `rule_based.py`           | Rule-based policy |
| `shallow_q.py`            | Q-learning agent  |
| `baseline_analysis.ipynb` | Full evaluation   |
| `plots/`                  | Figures           |
| `README.md`               | Documentation     |

---

# ▶️ How to Run

Open:

```
Baseline Policies/RuleBased_Shallow.ipynb
```

Running all cells will generate:

* Action distribution
* Learning curve
* Q-table heatmap
* Performance comparison tables


