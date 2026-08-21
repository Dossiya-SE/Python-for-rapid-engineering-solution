# Python for Rapid Engineering Solutions — EDA Evidence Archive

**EEE 591 · Dossiya Dakou**

<p align="center"><img src="assets/eda-mathematical-map.svg" width="100%" alt="EDA feature, covariance, correlation and evidence map" /></p>

## Generated evidence

<p align="center">
  <img src="assets/correlation_heatmap.png" width="31%" alt="Correlation heatmap" />
  <img src="assets/covariance_heatmap.png" width="31%" alt="Covariance heatmap" />
  <img src="assets/pairplot.png" width="31%" alt="Pair plot" />
</p>

The project studies multivariate structure in [`heart1.csv`](heart1.csv), including the declared target `a1p2`.

## Mathematical objects

```math
\Sigma=E[(X-\mu)(X-\mu)^T],
\qquad
\rho_{ij}=\frac{\Sigma_{ij}}{\sigma_i\sigma_j}.
```

Covariance is scale-sensitive. Correlation is used here as an **association diagnostic**, not evidence of causation.

## Evidence status

| Object | Status |
|---|---|
| [`heart1.csv`](heart1.csv) | committed dataset |
| [`Project1_Dossiya.pdf`](Project1_Dossiya.pdf) | committed report |
| three diagnostic figures | committed generated outputs |
| [`Problem1_of_Project1.py`](Problem1_of_Project1.py) | **placeholder code**, not the figure-generating analysis |
| Evidence-integrity workflow | automated artifact/schema check |

Current state: **artifact-verifiable but not yet source-reproducible** because the full generating analysis notebook/script is not publicly committed.

## Interpretation boundary

EDA can reveal association, redundancy, scale structure and unusual geometry. It does **not** establish causality, clinical validity, statistical significance, predictive generalization or a production model.

## Closure gate

`exact source → dependency specification → deterministic preprocessing → single regeneration command → column/schema checks → data provenance → output checks`

> A figure proves that an output exists. Reproducible source and scientific validity require additional evidence.
