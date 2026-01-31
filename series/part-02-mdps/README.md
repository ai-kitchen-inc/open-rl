# Part 02 — MDPs (Delayed Rewards & State)

**Concept focus:** States, transitions, and delayed rewards.

## What you’ll build
A tiny two-room gridworld with a door and a Monte Carlo value estimate.

## How to run
```bash
python train.py
python plot.py
```

## Expected output
- `outputs/state_values.npy`
- `outputs/state_values.png`

## Pitfalls + evaluation
- Short episodes hide delayed rewards.
- Random policy value can be noisy.
- Evaluate by averaging many episodes.
