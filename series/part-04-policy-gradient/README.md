# Part 04 — Policy Gradient (Learn the Policy Directly)

**Concept focus:** Directly learning action probabilities from reward.

## What you’ll build
A two-step decision chain trained with REINFORCE.

## How to run
```bash
python train.py
python plot.py
```

## Expected output
- `outputs/policy_probs.npy`
- `outputs/policy_probs.png`

## Pitfalls + evaluation
- High variance without a baseline.
- Learning rate too high can destabilize probabilities.
