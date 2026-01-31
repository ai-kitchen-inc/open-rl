# Part 06 — Model-Based RL (Planning)

**Concept focus:** Learn a model, then plan with it.

## What you’ll build
A Dyna-Q agent that uses simulated updates in a small grid.

## How to run
```bash
python train.py
python plot.py
```

## Expected output
- `outputs/model_free_rewards.npy`
- `outputs/dyna_rewards.npy`
- `outputs/planning_comparison.png`

## Pitfalls + evaluation
- Bad model estimates can mislead planning.
- Compare learning speed with/without planning.
