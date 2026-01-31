# Getting Started

This repo uses only NumPy + Matplotlib. The steps below get you running every tutorial quickly.

## 1) Create and activate a virtual environment (recommended)
```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 2) Install requirements
```bash
pip install -r requirements.txt
```

## 3) Run the main tutorial
```bash
cd tutorial
python train.py
python plot.py
```

Expected outputs live in `tutorial/outputs/`.

## 4) Run all series tutorials
From the repo root:
```bash
for part in series/part-*; do (cd "$part" && python train.py && python plot.py); done
```

Each part writes its artifacts into an `outputs/` folder inside that part.
