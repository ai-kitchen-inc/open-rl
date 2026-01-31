# Part 08 — Exploration (Efficient Discovery)

**Concept focus:** Explore intelligently, not randomly.

## What you’ll build
A sparse-reward grid with UCB-style exploration bonuses.

## How to run
```bash
python train.py
python plot.py
```

## Expected output
- `outputs/visit_counts.npy`
- `outputs/visits_heatmap.png`

## Pitfalls + evaluation
- Exploration bonus too high can prevent convergence.
- Track visitation coverage as well as reward.
