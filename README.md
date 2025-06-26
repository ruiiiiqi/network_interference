# Robust Inference for the Direct Average Treatment Effect with Treatment Assignment Interference

**Replication package** for the paper  
**[Matias D. Cattaneo, Yihan He, and Ruiqi (Rae) Yu (2025)](https://arxiv.org/abs/2502.13238).**

## Repository layout

| Path / script | Purpose |
|---------------|---------|
| `simulation_vary_temperature.py` | 5000 Monte Carlo replications across interaction strengths $\beta = 0, 0.1,\cdots,0.9, 0.95, 0.99, 0.995, 0.999, 0.9999, 1$; records interval length and empirical coverage |
| `simulation_vary_size.py` | 2000 replications across network sizes $n = 500, 1000, 2000, 4000, 8000, 16000$; records the *true* confidence interval length from Monte Carlo simulations, and the average conservative confidence interval length |
| `simulation_outputs/` | CSV files created by the two simulation scripts |
| `diagnosis.py` | Generates the diagnostic panels shown in Fig. 1 of the manuscript |

---

## Quick start

```bash

# 1  run simulations
python simulation_vary_temperature.py  # 5 000 replicates
python simulation_vary_size.py         # 2 000 replicates

# 2  plot diagnostics
python diagnosis.py
```

---

## Runtime & resources

These Monte Carlo experiments are **CPU-heavy**:

| Script | Replications | Typical wall-time on a MacBook Air M2 (8-core, 16 GB RAM, macOS Sequoia 15.5) |
|--------|--------------|-------------------------------------------------------------------------------|
| `simulation_vary_temperature.py` | 5 000 | ≈ 20 mins     |
| `simulation_vary_size.py`        | 2 000 | ≈ 9.5 hours   |

Both programs use Python’s `multiprocessing` module and will, by default, spin up **one worker per physical core** (8 on the M2 Air).  
You can override this by changing the second to last line `Pool(8) as p:` to `Pool(4) as p:` to use 4 cores, e.g.

