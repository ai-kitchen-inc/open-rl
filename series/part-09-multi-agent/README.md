# Part 09 — Multi-Agent (Coordination)

**Concept focus:** When multiple learners interact.

## What you’ll build
A coordination game where two agents must align actions.

## How to run
```bash
python train.py
python plot.py
```

## Expected output
- `outputs/joint_counts.npy`
- `outputs/joint_actions.png`

## Pitfalls + evaluation
- Non-stationarity: each agent changes the environment for the other.
- Track joint action distribution over time.
