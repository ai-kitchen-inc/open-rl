# Part 07 — Offline RL Intuition (Logged Data)

**Concept focus:** Learning from past data without new interaction.

## What you’ll build
A logged bandit dataset and a fitted Q evaluation estimate.

## How to run
```bash
python train.py
python plot.py
```

## Expected output
- `outputs/estimated_values.npy`
- `outputs/offline_eval.png`

## Pitfalls + evaluation
- Data bias: logs only cover certain actions.
- Evaluate with confidence intervals or multiple seeds.
