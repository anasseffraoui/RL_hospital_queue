# Hospital Queue Scheduling under Uncertainties

## Overview

This project addresses hospital queue management using Reinforcement Learning. The goal is to optimize patient scheduling by balancing three competing objectives:

- 40% Walk-in waiting times
- 40% Appointment compliance  
- 20% Throughput

We compared three RL algorithms (PPO, SAC, Q-Learning) and discovered that **problem reformulation trumps algorithm sophistication**.

---

## Problem Statement

### Challenge

A hospital system with:
- Multiple servers with heterogeneous capabilities
- Stochastic patient arrivals
- Mix of walk-ins and scheduled appointments
- Patient abandonment after excessive waits
- Critical appointments requiring ±3 min timing

### Decision Point

When a server becomes available: **Which waiting customer should be served next?**

### Evaluation Metric
```
Total Score = 0.4 × Gw + 0.4 × Ga + 0.2 × Gs
```

Where:
- **Gw** (0-100): Walk-in waiting time score
- **Ga** (0-100): Appointment compliance score
- **Gs** (0-100): Throughput score

---

## Project Structure
```
project/
├── app/
│   ├── data/
│   │   ├── breaks/              # Break schedules
│   │   ├── config/              # Configuration files
│   │   ├── data_files/          # Training data
│   │   ├── results/             # Evaluation results
│   │   ├── Instance.py
│   │   └── Scenario.py
│   ├── domain/
│   │   ├── Customer.py
│   │   ├── Server.py
│   │   ├── Task.py
│   │   └── Appointment.py
│   ├── simulation/
│   │   ├── activity/            # Server activities
│   │   ├── envs/
│   │   │   ├── Env.py          # Base environment
│   │   │   ├── ChildEnv.py     # RL environment (458 features)
│   │   │   └── RandomEnv.py
│   │   ├── events/             # Discrete event simulation
│   │   ├── policies/
│   │   │   ├── Policy.py       # Base policy
│   │   │   ├── ChildPolicy.py  # Q-Learning implementation
│   │   │   └── Random.py       # Random baseline
│   │   └── ...
│   ├── utils/
│   │   └── io_utils.py
│   ├── evaluate.py              # Evaluation script
│   └── main.py                  # Training script
├── instance_set/                # 50 test instances
│   ├── appointments_*.json
│   ├── timeline_*.json
│   └── unavailability_*.json
├── models/
│   └── Env/
│       ├── q_table.npy         # Trained Q-Learning model
│       └── q_meta.json         # Model metadata
├── results/                     # Output directory
└── requirements.txt
```

---

## What We Built

### RL Environment

**State Space (458 features):**
- Customer Features (50 × 9):
  - Waiting time, task compatibility, service time
  - Appointment status, urgency, delay
  - Critical window flag, abandonment risk
  - Exists flag (padding mask)
  
- Global Context (8 features):
  - Simulation time, queue size, server ID
  - Busy ratio, average wait, appointment pressure
  - Time remaining, served ratio

**Action Space (51 actions):**
- Actions 0-49: Serve customer i
- Action 50: HOLD (don't assign)
- Action masking prevents invalid selections

**Reward Function:**
- Aligned with evaluation (40-40-20 weights)
- Walk-ins: penalize long waits (linear up to 60 min)
- Appointments: perfect window at ±3 min
- Bonuses: +20 abandonment prevention, +10 perfect timing

---

## Algorithms Tested

### 1. PPO (Proximal Policy Optimization)

**Implementation:**
- On-policy actor-critic
- Neural networks: [128, 128] architecture
- MaskablePPO for native action masking
- Hybrid mode: ±5 min rule for critical appointments

**Results:**
- Total: 67.8 (+24.8 vs random baseline)
- Gw: 50.2, Ga: 85.3, Gs: 75.1
- Training: 1-3 hours

**Limitations:**
- Poor waiting time optimization
- High-dimensional space hard to explore
- Sample inefficient (on-policy)

### 2. SAC (Soft Actor-Critic)

**Implementation:**
- Off-policy with 100k replay buffer
- Maximum entropy objective
- Continuous-to-discrete wrapper

**Results:**
- Total: 68.0 (only +0.2 vs PPO)
- Gw: 55.4, Ga: 85.1, Gs: 77.3
- Training: 2-3 hours

**Why it failed:**
- Wrapper loses masking precision
- Appointment timing still too hard
- Entropy conflicts with sparse rewards

### 3. Q-Learning with Priority Weights

**Key Innovation:**
- Paradigm shift: Learn which **strategy** to use, not which **customer** to serve
- Two-stage decision:
  1. Q-Learning selects 1 of 18 weight configurations
  2. Priority formula automatically picks customer

**Priority Formula:**
```
Pi = w1·Ki + w2·Ti^1.5 + w3·Ui + w4·Ei
```

Where:
- Ki: Appointment urgency
- Ti^1.5: Waiting time (exponential scaling)
- Ui: Abandonment risk
- Ei: Service efficiency

**State Discretization:**
- 81 states (4 dimensions × 3 bins):
  - Appointment pressure: none / some / many
  - Queue pressure: light / medium / heavy
  - Abandonment risk: none / some / several
  - Time of day: morning / midday / afternoon

**Actions:**
- 18 weight configurations covering diverse strategies
- Examples: Balanced, Appt-focused, Wait-focused, Abandon-focused

**Results:**
- **Total: 84.6** (+16.8 vs PPO)
- Gw: 82.1, Ga: 88.2, Gs: 84.7
- Training: **17 minutes** (10× faster)
- Memory: <1 MB (200× smaller)
- Fully interpretable Q-table

---

## Results Comparison

| Metric | Random | PPO | SAC | Q-Learning |
|--------|--------|-----|-----|------------|
| **Total** | 43.0 | 67.8 | 68.0 | **84.6** |
| Gw (wait) | 35.1 | 50.2 | 55.4 | **82.1** |
| Ga (appt) | 60.3 | 85.3 | 85.1 | **88.2** |
| Gs (thput) | 60.2 | 75.1 | 77.3 | **84.7** |
| Training | - | 1-3h | 2-3h | **17 min** |
| Memory | <1MB | 200MB | 600MB | **<1MB** |
| Interpretable | Yes | No | No | **Yes** |

**Key Takeaway:** Simpler action space + domain knowledge > more powerful model

---

## Installation

### Prerequisites

Python 3.8 or higher

### Install Dependencies
```bash
pip install -r requirements.txt
```

Required packages:
```
gymnasium==0.29.0
numpy>=1.24.0
stable-baselines3==2.0.0
sb3-contrib==2.0.0
pandas>=2.0.0
matplotlib>=3.7.0
```

---

## How to Run

### Training Q-Learning
```bash
python app/main.py
```

This will:
- Initialize Q-Learning with 18 weight configurations
- Train for 500 episodes (~17 minutes)
- Save Q-table to `models/Env/q_table.npy`
- Print progress every 50 episodes

Expected output:
```
============================================================
Q-LEARNING WEIGHT OPTIMIZER
============================================================
States:  81 | Actions: 18
α=0.1, γ=0.9, Episodes=500
============================================================

Ep   50/500 | Avg: 12453.2 | Best: 14231.8 | ε: 0.778
Ep  100/500 | Avg: 13892.1 | Best: 15103.4 | ε: 0.605
...
Ep  500/500 | Avg: 16789.3 | Best: 18234.1 | ε: 0.082

============================================================
Training complete! Best reward: 18234.1
============================================================
```

### Evaluating the Model
```bash
python app/evaluate.py
```

This will:
- Load trained Q-table
- Test on 50 instances
- Compute Gw, Ga, Gs, and Total scores
- Save results to `app/data/results/result_0.csv`

Expected output:
```
============================================================
EVALUATING Q-LEARNING POLICY
============================================================
Testing on 50 instances...

Instance 0/50: Total=83.5
Instance 1/50: Total=85.9
...
Instance 49/50: Total=84.2

============================================================
FINAL RESULTS
============================================================
Gw (wait):         82.1
Ga (appointments): 88.2
Gs (throughput):   84.7
Total Score:       84.6
============================================================
```

## Understanding Q-Learning Approach

### The Paradigm Shift

Instead of:
```
458 features → Neural Network → Which of 51 customers?
```

We use: 
    Situation → Q-table → Which weights? → Priority formula → Customer

### Why Ti^1.5 Matters

Exponential waiting time scaling automatically prioritizes long waiters:

| Wait Time | Linear | Exponential (Ti^1.5) |
|-----------|--------|----------------------|
| 20 min | 0.33 | 0.19 |
| 40 min | 0.67 | 0.55 |
| 60 min | 1.00 | 1.00 |

Doubling wait from 20→40 min increases priority by **

# References 
Key Paper:
Adhicarna, I., Nurhidayati, S., & Fauzan, T. R. (2024). Optimization of Hospital Queue Management Using Priority Queue Algorithm and Reinforcement Learning. International Journal Software Engineering and Computer Science (IJSECS), 4(2), 512-522.

# Authors
Anas SEFFRAOUI
Email: anas.seffraoui@mines-ales.fr

Salmane EL HAJOUJI
Email: salmane.elhajouji@mines-ales.fr

Iljada BARDHOSHI
Email: iljada.bardhoshi@mines-ales.fr

Institution: IMT Mines Alès, Department of Computer Science and Artificial Intelligence
Date: February 2026

# License
This project is part of academic research at IMT Mines Alès.