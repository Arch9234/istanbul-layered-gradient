# -*- coding: utf-8 -*-
"""
================================================================================
Istanbul as a Layered Gradient - Full Analysis Pipeline
================================================================================

Reproducibility script for the manuscript:
    "Istanbul as a Layered Gradient: A Structural Analysis of Street-Scale
     Urban Fabric in a Sample Guided by Classical Urban Safety Theories"
    Submitted to Environment and Planning B: Urban Analytics and City Science
    (under peer review).

Pipeline:
    1.  Load per-street master dataset (74 streets)
    2.  Extract 12 retained features from semantic segmentation output
    3.  Standardize features and run K-means (k = 2..10)
    4.  Select k using silhouette + minimum cluster size validity criterion
    5.  Renumber cluster labels by ascending mean building-pixel share
        (deterministic labels; see Section 3.6 of the manuscript)
    6.  Statistical tests: chi-square + bias-corrected Cramer's V, Kruskal-
        Wallis with FDR, Spearman correlation, logistic regression
    7.  Five robustness tests: silhouette sweep, HDBSCAN noise, GMM, Ward-
        linkage cophenetic correlation, bootstrap consensus matrix
    8.  Sensitivity analysis: exclude Firuzkoy (n = 60), rerun everything

Random seed:
    All stochastic operations use random_state = 42 for exact reproducibility.

Requirements:
    Python 3.9+
    pandas, numpy, scipy, scikit-learn, hdbscan, statsmodels, openpyxl

Data format:
    Expects `data/istanbul_streets_master.xlsx` with a sheet named "streets"
    containing 74 rows and columns including:
        street_name_en, neighborhood, district, bilen_score, bilen_band,
        safety_class, and 19 semantic-segmentation columns prefixed with seg_.

Usage:
    pip install -r requirements.txt
    python 02_analysis_pipeline.py

Outputs are written to ./outputs/.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    log_loss,
)
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import pdist
from scipy.stats import (
    spearmanr,
    chi2_contingency,
    fisher_exact,
    kruskal,
)
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

try:
    import hdbscan
except ImportError:
    print("ERROR: hdbscan is required. Install with: pip install hdbscan")
    sys.exit(1)


# =============================================================================
# CONFIGURATION
# =============================================================================
RANDOM_STATE = 42
MANUAL_K = 3
MIN_PER_CLUSTER = 3
N_INIT_KMEANS = 20
N_BOOTSTRAP = 1000
BOOTSTRAP_STABILITY_THRESHOLD = 0.80

DATA_PATH = Path("data/istanbul_streets_master.xlsx")
DATA_SHEET = "streets"
OUTPUT_PATH = Path("outputs")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# 12 retained features (Section 3.4)
FEATURES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "vegetation", "terrain", "sky", "person", "car", "truck",
]


# =============================================================================
# DATA LOADING AND PREPARATION
# =============================================================================
def load_data(path=DATA_PATH, sheet=DATA_SHEET):
    if not path.exists():
        print(f"ERROR: Data file not found at {path.absolute()}")
        sys.exit(1)
    return pd.read_excel(path, sheet_name=sheet)


def prepare_data(df_raw):
    df = df_raw.copy()

    # Segmentation columns (with or without seg_ prefix)
    for feat in FEATURES:
        if feat not in df.columns:
            seg_col = f"seg_{feat}"
            if seg_col in df.columns:
                df[feat] = df[seg_col]
            else:
                print(f"ERROR: Feature column missing: {feat}")
                sys.exit(1)

    # Street column
    if "Street" not in df.columns:
        for cand in ["street_name_en", "street_name", "Sokak"]:
            if cand in df.columns:
                df["Street"] = df[cand].astype(str).str.strip()
                break

    # Neighborhood
    if "Neighborhood" not in df.columns and "neighborhood" in df.columns:
        df["Neighborhood"] = df["neighborhood"].astype(str).str.strip()

    # District
    if "District" not in df.columns and "district" in df.columns:
        df["District"] = df["district"].astype(str).str.strip()

    # Bilen score
    if "Bilen_Score" not in df.columns and "bilen_score" in df.columns:
        df["Bilen_Score"] = df["bilen_score"]

    # Safety label
    if "Safety_Label" not in df.columns:
        if "safety_class" in df.columns:
            df["Safety_Label"] = df["safety_class"].astype(str).str.strip().map(
                {"safe": "Safe", "Safe": "Safe",
                 "unsafe": "Unsafe", "Unsafe": "Unsafe"}
            )
        else:
            print("ERROR: No safety-label column found.")
            sys.exit(1)

    # Three-band safety category (Section 3.2)
    if "Safety_3Cat" not in df.columns:
        if "bilen_band" in df.columns:
            band_map = {"Low": "Low", "Middle": "Middle",
                        "Mid": "Middle", "High": "High"}
            df["Safety_3Cat"] = df["bilen_band"].astype(str).str.strip().map(
                lambda b: band_map.get(b, "Unknown")
            )

    print(f"Loaded : {len(df)} streets")
    print(f"Features : {len(FEATURES)}  (Section 3.4)")
    print(f"Safe : {(df['Safety_Label']=='Safe').sum()}  |  "
          f"Unsafe : {(df['Safety_Label']=='Unsafe').sum()}")
    if "Safety_3Cat" in df.columns:
        print(f"Low : {(df['Safety_3Cat']=='Low').sum()}  |  "
              f"Middle : {(df['Safety_3Cat']=='Middle').sum()}  |  "
              f"High : {(df['Safety_3Cat']=='High').sum()}")

    return df


# =============================================================================
# CLUSTER-LABEL RENUMBERING (Section 3.6)
# =============================================================================
def renumber_by_building_share(labels, df, feature_col="building"):
    tmp = df.copy()
    tmp["_cluster"] = labels
    order = tmp.groupby("_cluster")[feature_col].mean().sort_values().index.tolist()
    remap = {old: new for new, old in enumerate(order)}
    return np.array([remap[l] for l in labels])


# =============================================================================
# K SELECTION
# =============================================================================
def determine_k(X, manual_k=None, min_per_cluster=MIN_PER_CLUSTER,
                k_range=range(2, 11), label=""):
    print(f"\nSilhouette sweep (k = 2..10){' -- ' + label if label else ''}")
    print("-" * 60)
    scores = {}
    valid_ks = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=N_INIT_KMEANS,
                    random_state=RANDOM_STATE).fit(X)
        min_size = int(np.bincount(km.labels_).min())
        sil = silhouette_score(X, km.labels_)
        scores[k] = sil
        is_valid = min_size >= min_per_cluster
        if is_valid:
            valid_ks.append(k)
        tag = "OK" if is_valid else "invalid (min_size < 3)"
        print(f"  k = {k:2d}  |  silhouette = {sil:.4f}  |  "
              f"min cluster = {min_size:2d}  |  {tag}")

    if manual_k is not None:
        print(f"\n=> Using manual k = {manual_k}")
        return manual_k, scores

    valid_scores = {k: scores[k] for k in valid_ks}
    best_k = max(valid_scores, key=valid_scores.get)
    print(f"\n=> Auto-selected k = {best_k}")
    return best_k, scores


# =============================================================================
# CHI-SQUARE + BIAS-CORRECTED CRAMER'S V (Bergsma, 2013)
# =============================================================================
def bias_corrected_cramers_v(ct):
    ct = np.asarray(ct, dtype=float)
    n = ct.sum()
    r, c = ct.shape
    chi2, p, dof, _ = chi2_contingency(ct)
    phi2 = chi2 / n
    phi2_corr = max(0.0, phi2 - ((r - 1) * (c - 1)) / (n - 1))
    r_corr = r - (r - 1) ** 2 / (n - 1)
    c_corr = c - (c - 1) ** 2 / (n - 1)
    denom = min(r_corr - 1, c_corr - 1)
    v = float(np.sqrt(phi2_corr / denom)) if denom > 0 else np.nan
    return {"chi2": float(chi2), "p": float(p), "df": int(dof),
            "V": v, "n": int(n)}


def chi_square_cluster_safety(df, labels):
    ct = pd.crosstab(labels, df["Safety_Label"])
    print("\n" + "=" * 60)
    print("CHI-SQUARE / FISHER  --  Cluster x Safety_Label")
    print("=" * 60)
    print(ct)
    if ct.shape == (2, 2):
        or_val, p = fisher_exact(ct.values)
        print(f"\nFisher exact: OR = {or_val:.4f},  p = {p:.4f}")
        return {"contingency": ct, "OR": float(or_val), "p": float(p)}
    stats = bias_corrected_cramers_v(ct)
    print(f"\nchi-square = {stats['chi2']:.4f}  |  df = {stats['df']}  |  "
          f"p = {stats['p']:.4f}")
    print(f"Bias-corrected Cramer's V = {stats['V']:.4f} (Bergsma, 2013)")
    return {"contingency": ct, **stats}


# =============================================================================
# KRUSKAL-WALLIS + FDR
# =============================================================================
def kruskal_with_fdr(df, features):
    print("\n" + "=" * 60)
    print("KRUSKAL-WALLIS + FDR (Benjamini-Hochberg)")
    print("=" * 60)
    rows = []
    for feat in features:
        groups = [df[df["Safety_3Cat"] == c][feat].values
                  for c in ["Low", "Middle", "High"]
                  if c in df["Safety_3Cat"].values]
        if len(groups) >= 2:
            h, p_raw = kruskal(*groups)
            rows.append({"Feature": feat, "H": h, "p_raw": p_raw})
    df_res = pd.DataFrame(rows).sort_values("p_raw").reset_index(drop=True)
    if not df_res.empty:
        _, p_fdr, _, _ = multipletests(df_res["p_raw"].values, method="fdr_bh")
        df_res["p_fdr"] = p_fdr
        df_res["Significant"] = df_res["p_fdr"] < 0.05
    print(df_res.round(4).to_string(index=False))
    print(f"\n{df_res['Significant'].sum()}/{len(df_res)} features "
          f"significant after FDR")
    return df_res


# =============================================================================
# SPEARMAN CORRELATION AT NEIGHBORHOOD SCALE
# =============================================================================
def spearman_neighborhood(df, features):
    nbh_df = df.groupby(["Neighborhood", "Bilen_Score"])[features].mean().reset_index()
    print("\n" + "=" * 60)
    print(f"SPEARMAN  (neighborhood scale, n = {len(nbh_df)})")
    print("=" * 60)
    rows = []
    for feat in features:
        r, p = spearmanr(nbh_df[feat], nbh_df["Bilen_Score"])
        rows.append({"Feature": feat, "r": r, "p": p,
                     "Significant": p < 0.05})
    df_res = pd.DataFrame(rows).sort_values("r", ascending=False)
    print(df_res.round(4).to_string(index=False))
    return df_res, nbh_df


# =============================================================================
# LOGISTIC REGRESSION
# =============================================================================
def logistic_regression_bilen(df, features, threshold=None):
    nbh_df = df.groupby(["Neighborhood", "Bilen_Score"])[features].mean().reset_index()
    med = threshold if threshold is not None else nbh_df["Bilen_Score"].median()
    nbh_df["Safe_binary"] = (nbh_df["Bilen_Score"] >= med).astype(int)

    X = StandardScaler().fit_transform(nbh_df[features].values)
    y = nbh_df["Safe_binary"].values

    model = LogisticRegression(max_iter=1000, C=0.1, random_state=RANDOM_STATE)
    model.fit(X, y)
    y_prob = model.predict_proba(X)
    ll_full = -log_loss(y, y_prob, normalize=False)
    p_null = y.mean()
    ll_null = np.sum(y * np.log(p_null) + (1 - y) * np.log(1 - p_null))
    n = len(y)
    mcfadden = 1 - (ll_full / ll_null)
    cox_snell = 1 - np.exp((2 / n) * (ll_null - ll_full))
    nagelkerke = cox_snell / (1 - np.exp((2 / n) * ll_null))
    acc = float(np.mean(model.predict(X) == y))

    print("\n" + "=" * 60)
    print(f"LOGISTIC REGRESSION  (n = {n}, threshold = {med:.2f})")
    print("=" * 60)
    print(f"McFadden pseudo-R^2 : {mcfadden:.4f}")
    print(f"Cox-Snell R^2       : {cox_snell:.4f}")
    print(f"Nagelkerke R^2      : {nagelkerke:.4f}")
    print(f"Training accuracy    : {acc:.4f}")
    return {"mcfadden": mcfadden, "cox_snell": cox_snell,
            "nagelkerke": nagelkerke, "accuracy": acc, "n": n}


# =============================================================================
# ROBUSTNESS: HDBSCAN
# =============================================================================
def hdbscan_test(X, min_cluster_sizes=range(3, 8)):
    print("\n" + "=" * 60)
    print("HDBSCAN  --  min_cluster_size sweep")
    print("=" * 60)
    rows = []
    for mcs in min_cluster_sizes:
        lab = hdbscan.HDBSCAN(min_cluster_size=mcs).fit_predict(X)
        n_clusters = len(set(lab) - {-1})
        noise_pct = 100.0 * (lab == -1).mean()
        rows.append({"min_cluster_size": mcs, "n_clusters": n_clusters,
                     "noise_pct": noise_pct})
        print(f"  mcs = {mcs}: n_clusters = {n_clusters}, "
              f"noise = {noise_pct:.1f}%")
    return pd.DataFrame(rows)


# =============================================================================
# ROBUSTNESS: GMM
# =============================================================================
def gmm_test(X, k_range=range(2, 8)):
    print("\n" + "=" * 60)
    print("GAUSSIAN MIXTURE MODEL  --  BIC selection")
    print("=" * 60)
    rows = []
    for k in k_range:
        gmm = GaussianMixture(n_components=k, random_state=RANDOM_STATE,
                              n_init=5).fit(X)
        probs = gmm.predict_proba(X)
        uncertain_pct = 100.0 * (probs.max(axis=1) < 0.6).mean()
        rows.append({"k": k, "BIC": gmm.bic(X),
                     "uncertain_pct": uncertain_pct})
        print(f"  k = {k}: BIC = {gmm.bic(X):.2f}, "
              f"uncertain (P < 0.6) = {uncertain_pct:.1f}%")
    return pd.DataFrame(rows)


# =============================================================================
# ROBUSTNESS: COPHENETIC CORRELATION
# =============================================================================
def cophenetic_correlation(X):
    dists = pdist(X)
    Z = linkage(dists, method="ward")
    coph, _ = cophenet(Z, dists)
    print("\n" + "=" * 60)
    print("WARD-LINKAGE COPHENETIC CORRELATION (Sokal & Rohlf, 1962)")
    print("=" * 60)
    print(f"Cophenetic correlation coefficient = {coph:.4f}")
    return float(coph)


# =============================================================================
# ROBUSTNESS: CROSS-METHOD ARI
# =============================================================================
def cross_method_ari(X, k):
    km_labels = KMeans(n_clusters=k, n_init=N_INIT_KMEANS,
                       random_state=RANDOM_STATE).fit_predict(X)
    hdb_labels = hdbscan.HDBSCAN(min_cluster_size=3).fit_predict(X)
    gmm_labels = GaussianMixture(n_components=k, random_state=RANDOM_STATE,
                                 n_init=5).fit_predict(X)
    ward_labels = AgglomerativeClustering(n_clusters=k,
                                          linkage="ward").fit_predict(X)

    methods = {"K-means": km_labels, "HDBSCAN": hdb_labels,
               "GMM": gmm_labels, "Ward": ward_labels}
    names = list(methods.keys())
    mat = pd.DataFrame(np.nan, index=names, columns=names)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i < j:
                mat.loc[a, b] = adjusted_rand_score(methods[a], methods[b])

    print("\n" + "=" * 60)
    print("CROSS-METHOD ADJUSTED RAND INDEX")
    print("=" * 60)
    print(mat.round(3).to_string())
    upper = mat.values[np.triu_indices_from(mat.values, k=1)]
    upper = upper[~np.isnan(upper)]
    mean_ari = float(np.mean(upper))
    print(f"\nMean pairwise ARI = {mean_ari:.4f}")
    return mat, mean_ari


# =============================================================================
# ROBUSTNESS: BOOTSTRAP CONSENSUS
# =============================================================================
def bootstrap_consensus(X, k, n_iter=N_BOOTSTRAP,
                        threshold=BOOTSTRAP_STABILITY_THRESHOLD):
    print("\n" + "=" * 60)
    print(f"BOOTSTRAP CONSENSUS MATRIX  ({n_iter} iterations, k = {k})")
    print("=" * 60)
    n = X.shape[0]
    counts = np.zeros((n, n))
    coobs = np.zeros((n, n))
    rng = np.random.default_rng(RANDOM_STATE)

    for it in range(n_iter):
        idx = rng.choice(n, size=n, replace=True)
        km = KMeans(n_clusters=k, n_init=N_INIT_KMEANS,
                    random_state=RANDOM_STATE + it).fit(X[idx])
        labels_boot = km.labels_
        unique = np.unique(idx)
        for u in unique:
            for v in unique:
                coobs[u, v] += 1
                if labels_boot[idx == u][0] == labels_boot[idx == v][0]:
                    counts[u, v] += 1

    with np.errstate(divide="ignore", invalid="ignore"):
        consensus = np.where(coobs > 0, counts / coobs, 0.0)

    upper_idx = np.triu_indices(n, k=1)
    upper_vals = consensus[upper_idx]
    n_pairs = len(upper_vals)
    n_stable = int((upper_vals >= threshold).sum())
    pct_stable = 100.0 * n_stable / n_pairs
    mean_coa = float(np.mean(upper_vals))

    print(f"Total pairs analyzed         : {n_pairs}")
    print(f"Stable pairs (>= {threshold})  : {n_stable} ({pct_stable:.1f}%)")
    print(f"Mean pairwise co-assignment  : {mean_coa:.4f}")
    return consensus, pct_stable, mean_coa


# =============================================================================
# CLUSTER COMPOSITION REPORT
# =============================================================================
def print_cluster_composition(df, labels):
    tmp = df.copy()
    tmp["Cluster"] = labels
    print("\n" + "=" * 60)
    print("CLUSTER COMPOSITION")
    print("=" * 60)
    for c in sorted(set(labels)):
        sub = tmp[tmp["Cluster"] == c].sort_values(
            ["Safety_Label", "Neighborhood", "Street"]
        )
        n_safe = (sub["Safety_Label"] == "Safe").sum()
        n_unsafe = (sub["Safety_Label"] == "Unsafe").sum()
        mean_bld = sub["building"].mean()
        print(f"\n-- CLUSTER {c}  (n = {len(sub)}  |  "
              f"Safe = {n_safe}  |  Unsafe = {n_unsafe}  |  "
              f"mean building = {mean_bld:.1f}%) --")
        for _, row in sub.iterrows():
            print(f"  [{row['Safety_Label']:6}] "
                  f"{str(row['Street']):40} | "
                  f"{str(row.get('Neighborhood', '?'))}")


# =============================================================================
# MAIN ANALYSIS BLOCK
# =============================================================================
def run_analysis(df_raw, label="MAIN"):
    print("\n" + "#" * 70)
    print(f"# {label} ANALYSIS")
    print("#" * 70)

    df = prepare_data(df_raw)

    X = df[FEATURES].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    best_k, sil_scores = determine_k(X_scaled, manual_k=MANUAL_K, label=label)

    km = KMeans(n_clusters=best_k, n_init=N_INIT_KMEANS,
                random_state=RANDOM_STATE).fit(X_scaled)
    labels = renumber_by_building_share(km.labels_, df)

    pca = PCA(n_components=2, random_state=RANDOM_STATE).fit(X_scaled)
    ev = pca.explained_variance_ratio_
    print(f"\nPCA explained variance: "
          f"PC1 = {ev[0]:.4f} ({ev[0]*100:.1f}%)  |  "
          f"PC2 = {ev[1]:.4f} ({ev[1]*100:.1f}%)")

    print_cluster_composition(df, labels)
    chi_res = chi_square_cluster_safety(df, labels)
    kw_res = kruskal_with_fdr(df, FEATURES)
    sp_res, nbh_df = spearman_neighborhood(df, FEATURES)
    lr_res = logistic_regression_bilen(df, FEATURES)

    print("\n" + "*" * 70)
    print("* ROBUSTNESS TESTS")
    print("*" * 70)
    coph = cophenetic_correlation(X_scaled)
    hdb_df = hdbscan_test(X_scaled)
    gmm_df = gmm_test(X_scaled)
    ari_matrix, mean_ari = cross_method_ari(X_scaled, best_k)
    consensus, pct_stable, mean_coa = bootstrap_consensus(X_scaled, best_k)

    df_out = df.copy()
    df_out["Cluster"] = labels
    out_cols = ["Street", "Neighborhood", "District", "Bilen_Score",
                "Safety_Label", "Safety_3Cat", "Cluster"]
    out_cols = [c for c in out_cols if c in df_out.columns]
    excel_path = OUTPUT_PATH / f"cluster_results_{label.lower()}.xlsx"
    df_out[out_cols].sort_values(
        ["Cluster", "Safety_Label", "Neighborhood"]
    ).to_excel(excel_path, index=False)
    print(f"\nCluster results saved: {excel_path}")

    return {
        "df": df_out, "labels": labels, "features": FEATURES,
        "best_k": best_k, "silhouette_scores": sil_scores,
        "silhouette_selected": sil_scores[best_k], "pca_explained": ev,
        "chi_square": chi_res, "kruskal": kw_res, "spearman": sp_res,
        "logistic": lr_res, "cophenetic": coph, "hdbscan": hdb_df,
        "gmm": gmm_df, "ari_matrix": ari_matrix, "mean_ari": mean_ari,
        "bootstrap_consensus": consensus,
        "bootstrap_stable_pct": pct_stable,
        "bootstrap_mean_coa": mean_coa,
    }


# =============================================================================
# SENSITIVITY ANALYSIS
# =============================================================================
def run_sensitivity(df_raw, exclude="Firuzkoy"):
    print("\n" + "#" * 70)
    print(f"# SENSITIVITY ANALYSIS  --  excluding {exclude}")
    print("#" * 70)

    col = "neighborhood" if "neighborhood" in df_raw.columns else "Neighborhood"

    # Match both ASCII and Turkish spellings of Firuzkoy
    mask = ~df_raw[col].astype(str).str.contains(
        "Firuzkoy|Firüzköy", regex=True, na=False
    )
    df_sens_raw = df_raw[mask].reset_index(drop=True)
    n_excluded = (~mask).sum()
    print(f"\nExcluded {n_excluded} streets. n = {mask.sum()}")

    return run_analysis(df_sens_raw, label="SENSITIVITY")


# =============================================================================
# SUMMARY
# =============================================================================
def print_summary(rm, rs):
    print("\n" + "=" * 78)
    print("SUMMARY  --  MAIN vs SENSITIVITY")
    print("=" * 78)
    fmt = "{:<38}{:>18}{:>22}"
    print(fmt.format("Metric", "Main (n=74)", "Sensitivity (n=60)"))
    print("-" * 78)
    print(fmt.format("Selected k",
                     str(rm['best_k']), str(rs['best_k'])))
    print(fmt.format("Silhouette",
                     f"{rm['silhouette_selected']:.4f}",
                     f"{rs['silhouette_selected']:.4f}"))
    print(fmt.format("Chi-square (Cluster x Safety)",
                     f"{rm['chi_square']['chi2']:.4f}",
                     f"{rs['chi_square']['chi2']:.4f}"))
    print(fmt.format("Bias-corrected Cramer's V",
                     f"{rm['chi_square']['V']:.4f}",
                     f"{rs['chi_square']['V']:.4f}"))
    print(fmt.format("Cophenetic correlation",
                     f"{rm['cophenetic']:.4f}",
                     f"{rs['cophenetic']:.4f}"))
    print(fmt.format("Mean pairwise ARI",
                     f"{rm['mean_ari']:.4f}",
                     f"{rs['mean_ari']:.4f}"))
    print(fmt.format("Bootstrap stable-pair %",
                     f"{rm['bootstrap_stable_pct']:.1f}%",
                     f"{rs['bootstrap_stable_pct']:.1f}%"))
    print("=" * 78)


# =============================================================================
# ENTRY POINT
# =============================================================================
def main():
    print("=" * 70)
    print("Istanbul as a Layered Gradient  --  Full Analysis Pipeline")
    print("=" * 70)
    print(f"Data path : {DATA_PATH.absolute()}")
    print(f"Sheet     : {DATA_SHEET}")
    print(f"Random    : random_state = {RANDOM_STATE}")
    print(f"Output    : {OUTPUT_PATH.absolute()}")

    df_raw = load_data()

    results_main = run_analysis(df_raw, label="MAIN")
    results_sens = run_sensitivity(df_raw)

    print_summary(results_main, results_sens)

    print("\nAll analyses complete. See ./outputs/ for saved files.")


if __name__ == "__main__":
    main()
