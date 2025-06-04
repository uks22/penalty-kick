
# ⚽ Strategic Analysis of Penalty Kicks in Football  
### Reinforcement Learning Meets Game Theory

> A novel framework combining **Reinforcement Learning (RL)** and **Game Theory** to simulate and analyze optimal penalty kick strategies in football.

---

## 🧠 Overview

This project investigates the strategic decision-making involved in football penalty kicks using two complementary approaches:
- **Game Theoretic modeling** of kicker vs goalkeeper as a zero-sum game.
- **Reinforcement Learning** to empirically train agents and discover effective kicking strategies.

Both **1D** and **2D goalpost environments** are explored, and goalkeeper skill profiles are learned using **real-world FIFA data (1982–2018)**.

---

## 📊 Core Contributions

- **Custom Reward Functions** using Gamma and Bivariate Gaussian distributions.
- **Game-Theoretic Equilibria** for both pure and mixed strategies.
- **RL Environment** for both 1D and 2D penalty kick scenarios.
- **Empirical Evaluation** with 500 synthetic goalkeepers and 50 real ones from FIFA data.
- **Heatmap Visualizations** for kicker and goalkeeper probability distributions.

---

## 🏗️ Methodology

### ✅ 1D Model
- Goalpost simplified to `[0, 20]` (valid region: `[5, 15]`).
- Kicker aims at a point `x` ∈ `[0, 20]`.
- Goalkeeper chooses from `{Left, Middle, Right}`.
- **Reward Function**:  
  \[
  f(x) = \beta \cdot x^\alpha \cdot e^{- \gamma x}
  \]

### ✅ 2D Model
- Goalpost defined on the `xy`-plane: `[0,10] x [0,10]`.
- Four target zones: UL, UR, BL, BR.
- **Reward Function**:  
  \[
  f(x, y) = \alpha \cdot e^{-(\beta x^2 + \gamma xy + \delta y^2)}
  \]

### ✅ Game-Theoretic Strategy
- Models Kicker vs Goalkeeper as a **2-player zero-sum game**.
- Optimal mixed strategies derived analytically:
  \[
  x_i = \frac{1/p_i}{\sum 1/p_j}
  \]

---

## 🧪 Experiments

- Trained RL agents against:
  - **500 dummy goalkeepers** (synthetic skill profiles).
  - **50 real goalkeepers** (from FIFA dataset).
- Evaluated:
  - Kicker’s aim distribution (for both 1D & 2D models).
  - Heatmaps of region preferences.
- Findings:
  - Kicker strategy **varies per goalkeeper**.
  - No universal best region; players tend to spread shots **uniformly**.

---

## 📁 Project Structure

\`\`\`bash
penalty-kick-strategy/
├── data/                      # Preprocessed FIFA dataset
├── models/                    # Trained RL models
├── environments/
│   ├── kicker_env_1d.py       # 1D goalpost simulation
│   └── kicker_env_2d.py       # 2D goalpost simulation
├── analysis/
│   ├── heatmaps.py            # Visualization scripts
│   └── mixed_strategy_solver.py
├── results/
│   └── figures/               # Heatmaps and probability graphs
├── paper/                     # Research paper PDF
└── README.md
\`\`\`

---

## 📦 Dataset

- Based on: [FIFA World Cup Penalty Kick Dataset (1982–2018)](https://www.kaggle.com/datasets/pablollanderos33/world-cup-penalty-shootouts)
- Extended with goalkeeper-wise save probabilities in 4 zones.

---

## 📉 Sample Output

- 🎯 Kicker Probability Distribution (1D and 2D)
- 🧱 Goalkeeper Skill Matrices
- 🔥 Heatmaps showing shot tendencies

---

## 📚 References

1. González Acuña & González Romero, 2018 — *Optimal Strategies for Penalty Kicks*
2. Palacios-Huerta, 2003 & 2023 — *Professionals Play Minimax*
3. Ekhosuehi, 2018 — *Penalty Kick Zero-Sum Game*
4. Tuyls et al., 2021 — *Game Plan: What AI Can Do for Football*
5. FIFA Penalty Kick Dataset — Kaggle

---

## 🚀 Future Work

- Use advanced RL methods (PPO, DDPG) for real-time feedback learning.
- Integrate psychological modeling (e.g., kicker confidence under pressure).
- Extend to goalkeeper strategy learning using adversarial RL.


