## Requirements

Python 3.9 or later. Install dependencies with:

```bash
pip install -r requirements.txt
```

Key libraries: `pandas`, `numpy`, `scipy`, `scikit-learn`, `hdbscan`,
`statsmodels`, `scikit-posthocs` (for DSCF post-hoc), `matplotlib`,
`seaborn`, `openpyxl`.

## Reproducing the analysis

### 1. Data collection (optional)

The `01_data_collection.py` script reproduces the Street View
download using the Google Street View Static API. It requires a
valid Google API key set as an environment variable
(`GOOGLE_STREETVIEW_API_KEY`) and a CSV of street start/end
coordinates in `./data/streets_coords.csv`.

Panoramas themselves are not redistributed in accordance with
Google's Terms of Service.

### 2. Run the analysis pipeline

The dataset (`data/istanbul_streets_master.xlsx`) provided in this
repository contains 74 streets with semantic-segmentation pixel
shares, safety labels, neighborhood-level Bilen scores, and safety
band assignments resulting from the human-supervised quality-
control step described in the manuscript (Section 3.5).

```bash
python 02_analysis_pipeline.py
```

Outputs (cluster assignments, DSCF post-hoc tables) are written to
`./outputs/`.

All stochastic operations use `random_state = 42`. Rerunning the
pipeline reproduces the deterministic statistics reported in the
manuscript exactly (silhouette, chi-square, Cramér's V, cluster
sizes, Kruskal-Wallis H, DSCF pairwise p-values, cophenetic,
HDBSCAN noise). Monte Carlo procedures (mean pairwise ARI and
bootstrap consensus stability) may vary by ±3% across runs due to
their stochastic nature.

### Note on figures

The manuscript's figures (Figures 1–5 in the main text, Figures
S6–S11 in the Supplementary Material) were generated from the
outputs of this pipeline using matplotlib. Because the underlying
data and statistics are fully reproducible via
`02_analysis_pipeline.py`, the visualization code is not included
in this repository. Readers wishing to regenerate the figures can
do so from the cross-tabulations produced by the pipeline; the
cluster × safety band distribution (Figure S8) relies directly on
the outputs of the main analysis block. The correspondence
analysis (Figure 4 area) was performed in SPSS using the input
file exported by the pipeline (`SPSS_prep.csv`, containing cluster
assignments and safety-band labels).

## Data note

- **Neighborhood-level safety scores** were derived from Bilen
  (2016), a publicly available doctoral thesis.
- **Original Google Street View panoramas** are not redistributed
  because of Google's terms of service.
- **Per-street pixel-share vectors and safety labels** (the
  analysis-ready dataset) are included in
  `data/istanbul_streets_master.xlsx`.
- **Binary safety labels (safety_class column):** Streets are
  labeled "Safe" if their neighborhood's Bilen score exceeds 17.55
  (median threshold, per Section 3.1 of the manuscript) and
  "Unsafe" otherwise. Firüzköy (score = 17.55) is classified as
  Unsafe.
- **Safety band assignment (Section 3.2 of the manuscript):**
  Firüzköy (bilen_score = 17.55) is grouped with Middle following
  the manuscript's methodological choice; the mapping is
  implemented in `NEIGHBORHOOD_BAND` in `02_analysis_pipeline.py`.

## License

Analysis code is distributed under the MIT License (see `LICENSE`).
Bilen (2016) safety-score data is used under fair academic use;
please cite the original thesis.

## Expected outputs

The pipeline reports and exports the following results, which match
the values presented in the manuscript:

### Main analysis (n = 74)

- Silhouette peak: **0.207** (k = 3)
- PCA (visualization only): PC1 + PC2 = 39.3% of variance
- Chi-square (Cluster × Safety label): **14.71** (p < 0.001);
  bias-corrected Cramér's V = **0.417**
- Cophenetic correlation (Ward linkage): **0.382**
- HDBSCAN noise ratio: **33.8%**
- Mean pairwise ARI (K-means, HDBSCAN, GMM, Ward): **0.261**
- Bootstrap consensus (1,000 K-means runs, threshold 0.80):
  675 / 2,701 stable pairs (**25.0%**)
- Correspondence analysis (Cluster × 3-band safety):
  χ² = 27.02 (df = 4, p < 0.001); Dimension 1 = 92.3%,
  Dimension 2 = 7.7% of inertia.

### Sensitivity analysis (Firuzkoy excluded, n = 60)

- Silhouette (k = 3): **0.156**
- Chi-square (Cluster × Safety): **3.73** (p = 0.155)
- Cramér's V (bias-corrected): **0.169**
- Cophenetic correlation: **0.571**
- HDBSCAN noise ratio: **76.7%**
- Mean ARI (KM–GMM–Ward, HDBSCAN excluded due to > 75% noise):
  **0.253**
- Bootstrap stability ≥ 0.80: 348 / 1,770 (**19.7%**)
- Cluster sizes: 24 / 31 / 5

## References

- **Bergsma W (2013)** — for bias-corrected Cramér's V. DOI:
  https://doi.org/10.1016/j.jkss.2012.10.002
- **Benjamini Y and Hochberg Y (1995)** — for FDR correction. DOI:
  https://doi.org/10.1111/j.2517-6161.1995.tb02031.x
- **Critchlow DE and Fligner MA (1991)** — for DSCF post-hoc
  pairwise tests. DOI:
  https://doi.org/10.1080/03610929108830487
- **van Dam A et al. (2021)** — for correspondence analysis. DOI:
  https://doi.org/10.1038/s41598-021-87971-9
- **Bilen O (2016)** — Istanbul safety index. PhD thesis, Yildiz
  Technical University.

---

## Citation

Citation details will be added upon acceptance.
