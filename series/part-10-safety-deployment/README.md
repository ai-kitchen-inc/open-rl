# Part 10 — Safety & Deployment (Guardrails)

**Concept focus:** Penalize unsafe behavior and monitor violations.

## What you’ll build
A safe gridworld with penalty-based constraints.

## How to run
```bash
python train.py
python plot.py
```

## Expected output
- `outputs/rewards.npy`
- `outputs/violations.npy`
- `outputs/safety_tradeoff.png`

## Pitfalls + evaluation
- Over-penalizing can stop exploration entirely.
- Track reward and safety violations together.
