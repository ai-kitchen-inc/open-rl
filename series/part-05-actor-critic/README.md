# Part 05 — Actor-Critic (Policy + Value)

**Concept focus:** Using a value function to stabilize policy learning.

## What you’ll build
A small gridworld trained with a tabular actor-critic.

## How to run
```bash
python train.py
python plot.py
```

## Expected output
- `outputs/rewards.npy`
- `outputs/learning_curve.png`

## Pitfalls + evaluation
- If the critic is wrong, the actor learns bad habits.
- Keep learning rates small and stable.
