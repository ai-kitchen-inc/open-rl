# Part 03 — Q-Learning (Value Learning)

**Concept focus:** Learning action values directly from experience.

## What you’ll build
A gridworld agent that learns to reach a goal while avoiding traps.

## How to run
```bash
python train.py
python plot.py
```

## Expected output
- `outputs/train_rewards.npy`
- `outputs/learning_curve.png`
- `outputs/policy.png`

## Pitfalls + evaluation
- High epsilon → noisy learning.
- Low epsilon → stuck in a local loop.
- Evaluate with greedy rollouts.
