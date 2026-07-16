# Istanbul as a Layered Gradient

Reproducibility materials for the manuscript:

> **"Istanbul as a Layered Gradient: A Structural Analysis of Street-Scale
> Urban Fabric in a Sample Guided by Classical Urban Safety Theories"**
>
> Currently under peer review at *Environment and Planning B: Urban
> Analytics and City Science*.

---

## Overview

This repository contains the full analysis pipeline for a semantic-segmentation
and clustering study of 74 streets in Istanbul, selected through
theory-informed purposive sampling from Bilen (2016) safety-index
neighborhoods.

The pipeline covers:

1. **Data collection** — Google Street View Static API, 12 sampling points × 12 headings per street.
2. **Semantic segmentation** — DeepLabv3+ (Xception-65, Cityscapes pre-trained), 12 retained pixel-share features.
3. **Clustering** — K-means (k = 3), cluster labels renumbered by ascending mean building-pixel share.
4. **Statistical tests** — chi-square with bias-corrected Cramér's V (Bergsma, 2013), Kruskal-Wallis with FDR correction, Spearman correlation at neighborhood scale, logistic regression (McFadden / Nagelkerke / Cox-Snell).
5. **Five robustness tests** — silhouette sweep, HDBSCAN noise, GMM assignment uncertainty, Ward-linkage cophenetic correlation, bootstrap consensus matrix (1000 iterations, threshold ≥ 0.80).
6. **Sensitivity analysis** — full pipeline rerun excluding Firüzköy (n = 60).

---

## Repository structure

```
.
├── 01_data_collection.py       # Google Street View download
├── 02_analysis_pipeline.py     # Full analysis and robustness tests
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT
├── README.md
└── data/
    └── istanbul_streets_master.xlsx    # 74 streets, semantic segmentation + safety labels
```

---

## Requirements

- Python 3.9 or later
- See `requirements.txt` for the full list.

Key libraries: `pandas`, `numpy`, `scipy`, `scikit-learn`, `hdbscan`,
`matplotlib`, `seaborn`, `statsmodels`, `openpyxl`.

For the data-collection step only: `googlemaps`, `requests`, `Pillow`.

---

## Reproducing the analysis

### 1. Set up the environment

```bash
pip install -r requirements.txt
```

### 2. Run the analysis pipeline

The dataset (`data/pixel_shares.xlsx`) provided in this repository contains
the 74 streets × 12 features matrix that resulted from the human-supervised
quality-control step described in the manuscript (Section 3.5).

```bash
python 02_analysis_pipeline.py
```

Outputs (Excel + PNG) are written to `./outputs/`.

All stochastic operations use `random_state = 42`. Rerunning the pipeline
produces the exact values reported in the manuscript.

### 3. (Optional) Reproduce the data collection

If you want to regenerate the raw Street View panoramas, set your Google
Maps Static API key as an environment variable:

```bash
export GOOGLE_MAPS_API_KEY=your_key_here
python 01_data_collection.py
```

**Note.** Original Google Street View panoramas are not redistributed here
because of Google's terms of service. The data-collection script allows
users with a valid API key to reproduce the raw image set.



## Expected outputs (verification)

Running `02_analysis_pipeline.py` on the provided dataset should yield:

**Main analysis (n = 74)**

| Metric | Value |
|---|---|
| Selected k | 3 |
| Silhouette | 0.207 |
| PCA explained variance (PC1 + PC2) | 39.3% |
| χ² (Cluster × Safety_Label) | 14.71 (p < 0.001) |
| Bias-corrected Cramér's V | 0.417 |
| Cophenetic correlation | 0.382 |
| HDBSCAN noise (min_cluster_size = 3) | 33.8% |
| Mean cross-method ARI | 0.261 |
| Bootstrap stability (pairs ≥ 0.80) | 25.0% |

**Sensitivity analysis (n = 60, Firüzköy excluded)**

| Metric | Value |
|---|---|
| Selected k | 3 |
| Silhouette | 0.156 |
| χ² (Cluster × Safety_Label) | 3.73 (p = 0.155) |
| Bias-corrected Cramér's V | 0.169 |
| HDBSCAN noise | 76.7% |
| Mean cross-method ARI | 0.164 |
| Bootstrap stability | 19.7% |

Full comparison is written to `outputs/cluster_results_main.xlsx` and
`outputs/cluster_results_sensitivity.xlsx`.

---

## Data note

- **Neighborhood-level safety scores** were derived from Bilen (2016),
  a publicly available doctoral thesis.
- **Original Google Street View panoramas** are not redistributed because
  of Google's terms of service.
- **Per-street pixel-share vectors** (the analysis-ready dataset) are
  included in `data/pixel_shares.xlsx`.

---

## License

- **Code**: MIT License (see `LICENSE`).
- **Data**: CC-BY 4.0, except Google Street View imagery which remains
  subject to Google's terms of service.

---

## Citation

Citation details will be added upon acceptance.
