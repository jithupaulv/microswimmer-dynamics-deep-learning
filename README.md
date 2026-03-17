# Magnetic Microswimmer Dynamics with LSTM, PINN, and Baseline Comparisons

A GitHub-ready research project that turns the reduced magnetic microswimmer dynamics into a machine-learning benchmark for:

- **summary prediction** from simulation data
- **sequence forecasting** with an **LSTM**
- **physics-constrained sequence learning** with a **PINN**
- **baseline comparisons** against an MLP surrogate

The project is based on the reduced nondimensional microswimmer dynamics in Eq. (4), the small-angle reduced ODE in Eq. (8), the harmonic-balance discussion around Eqs. (10)–(24), and the mean-speed expression in Eqs. (25)–(27). The paper studies the state variables \(\theta(t)\) and \(\phi(t)\), their periodic solutions, bifurcation behavior, and mean swimming speed as functions of the nondimensional parameters \(\beta\), \(\gamma\), and \(\omega\). fileciteturn2file0L61-L73 fileciteturn2file1L97-L130 fileciteturn2file1L159-L186

## What this repo does

1. **Simulates the reduced microswimmer dynamics** over many parameter settings.
2. **Builds supervised datasets** for:
   - regime classification
   - mean-speed regression
   - trajectory forecasting
3. **Trains three model families**:
   - **MLP baseline** on summary targets
   - **LSTM** for trajectory prediction
   - **PINN** for trajectory prediction with ODE residual loss
4. **Compares models** on held-out data.
5. **Generates publication-style plots**.

## Repo structure

```text
microswimmer_dl_project/
├── README.md
├── requirements.txt
├── configs/
│   └── default.yaml
├── scripts/
│   ├── generate_dataset.py
│   ├── train_baseline.py
│   ├── train_lstm.py
│   ├── train_pinn.py
│   ├── compare_models.py
│   └── plot_examples.py
└── src/
    └── microswimmer/
        ├── __init__.py
        ├── dynamics.py
        ├── data.py
        ├── models.py
        ├── train_utils.py
        └── utils.py
```

## Scientific setup

The reduced dynamics evolve in the two-state vector \(z(t) = (\theta(t), \phi(t))^T\), and the paper emphasizes that the dynamics depend on the dimensionless parameters \(\beta\), \(\gamma\), and \(\omega\). It also studies coexisting forward and backward periodic branches, stability transitions, and mean speed in the \(x\)-direction. fileciteturn2file1L61-L78 fileciteturn2file1L90-L99 fileciteturn2file1L159-L186

This repository uses:

- **Eq. (4)** for numerical trajectory generation
- **Eq. (8)** as a compact physics residual for PINN training
- **Eq. (25)** for numerical instantaneous speed
- **Eq. (27)** as inspiration for mean-speed evaluation and asymptotic comparisons

Those equations appear in the paper’s reduced-model and asymptotic analysis. fileciteturn2file0L61-L73 fileciteturn2file1L97-L130 fileciteturn2file1L159-L186

## Quickstart

### 1) Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
```

### 2) Generate data

```bash
python scripts/generate_dataset.py \
  --out_dir data/sim \
  --n_trajectories 400 \
  --t_final 15.0 \
  --dt 0.02
```

### 3) Train the MLP baseline

```bash
python scripts/train_baseline.py \
  --data_dir data/sim \
  --out_dir results/baseline
```

### 4) Train the LSTM sequence model

```bash
python scripts/train_lstm.py \
  --data_dir data/sim \
  --out_dir results/lstm
```

### 5) Train the PINN

```bash
python scripts/train_pinn.py \
  --data_dir data/sim \
  --out_dir results/pinn
```

### 6) Compare results

```bash
python scripts/compare_models.py \
  --data_dir data/sim \
  --baseline_dir results/baseline \
  --lstm_dir results/lstm \
  --pinn_dir results/pinn \
  --out_dir results/comparison
```

## Targets and metrics

### Baseline MLP
Input:
- \(\beta, \gamma, \omega, \theta_0, \phi_0\)

Outputs:
- mean \(\theta\)
- mean \(\phi\)
- mean speed
- regime label (forward/backward)

Metrics:
- MSE / MAE for regression
- accuracy for classification

### LSTM
Input:
- parameter token \((\beta,\gamma,\omega)\)
- short warmup segment of \((\theta,\phi)\)

Output:
- full future trajectory

Metrics:
- rollout MSE
- final-horizon error
- speed error after decoded rollout

### PINN
Input:
- time \(t\), parameters \((\beta,\gamma,\omega)\), initial condition \((\theta_0,\phi_0)\)

Output:
- \(\theta(t), \phi(t)\)

Loss:
- data loss
- initial-condition loss
- reduced-ODE residual loss from Eq. (4)
- optional compact residual from Eq. (8)

## Why this project is strong on GitHub

- clean scientific simulation pipeline
- real sequence modeling with an LSTM
- a PINN with explicit residual constraints
- clear baseline comparisons
- easy extension to bifurcation maps and stability diagrams

## Suggested extensions

- transformer sequence model
- neural operator for parameter-to-trajectory learning
- basin-of-attraction maps
- stability classifier near the pitchfork transition
- reproducing the paper’s speed and bifurcation curves more directly

## Citation

If you use the underlying dynamics in academic work, cite the original paper.
