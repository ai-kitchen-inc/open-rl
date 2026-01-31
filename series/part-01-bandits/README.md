# Part 01 — Bandits (Exploration vs. Exploitation)

**Concept focus:** When to try new options vs. stick with what works.

## What you’ll build
A 3-armed bandit learner using epsilon-greedy.

## How to run
```bash
python train.py
python plot.py
```

## Expected output
- `outputs/rewards.npy`
- `outputs/avg_reward.png`

## Pitfalls + evaluation
- Too much exploration → slow learning.
- Too little exploration → stuck on a suboptimal arm.
- Evaluate with average reward over time, not single episodes.
