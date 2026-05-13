# Kuka Anomaly Detection — AM01

Code for project **2026/AM01 — Detection of Anomalous Behaviour in Industrial Robot**
(MLinAPP, Politecnico di Torino, Project Owner: Alessio Mascolini).

The dataset (`../KukaVelocityDataset/`) consists of multivariate time-series sampled
from a KUKA industrial robot:

- `KukaNormal.npy` — `(233_792, 86)` — nominal recordings (no `anomaly` column).
- `KukaSlow.npy` — `(41_538, 87)` — anomalous recordings, robot moving slower than
  normal; the last column is the `anomaly` label (always 1).
- `KukaColumnNames.npy` — `(87,)` — names of all 87 columns
  (`action` first, `anomaly` last).

The 86 sensor channels = 1 `action` index + 8 robot-level electrical signals
(apparent power, current, frequency, phase angle, power, power factor, reactive
power, voltage) + 7 IMUs × 11 channels each (accel XYZ, gyro XYZ, quaternion
q1–q4, temperature).

## Layout

```
code/
├── README.md
├── requirements.txt
├── src/                  # importable modules
│   ├── data.py           # load and align the .npy arrays
│   ├── stats.py          # per-channel statistics
│   ├── viz.py            # plotting helpers
│   ├── separability.py   # PCA / UMAP / logistic-regression sanity check
│   └── utils.py          # device selection (MPS / CUDA / CPU), seeding
├── notebooks/            # exploratory Jupyter notebooks (kept thin)
├── configs/              # YAML configs for experiments (filled in later)
├── results/              # generated artifacts (figures, CSVs, markdown)
│   └── figures/
└── scripts/
    └── run_exploration.py  # end-to-end exploration entry point
```

## Setup

```bash
cd code/
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run the exploration

```bash
python scripts/run_exploration.py
```

This loads the dataset from `../KukaVelocityDataset/`, computes statistics,
generates figures into `results/figures/`, runs the separability sanity check,
and writes `results/EXPLORATION.md`.
