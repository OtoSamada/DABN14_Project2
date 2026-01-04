from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
from tqdm.auto import tqdm
from sklearn.base import ClusterMixin
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, OneHotEncoder, RobustScaler, StandardScaler, MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA


try:
    import hdbscan  # type: ignore
except ImportError:
    hdbscan = None  # Optional dependency

# -------------------------------------------------------------------
# Metrics
# -------------------------------------------------------------------
def hopkins_statistic(X: np.ndarray, m: Optional[int] = None, random_state: int = 42) -> float:
    X = np.asarray(X)
    n, d = X.shape
    if n < 3:
        return np.nan
    rng = np.random.default_rng(random_state)
    if m is None:
        m = int(min(max(10, 0.1 * n), 500))
    m = int(min(max(1, m), n - 1))
    nn = NearestNeighbors(n_neighbors=2).fit(X)
    sample_idx = rng.choice(n, size=m, replace=False)
    w = nn.kneighbors(X[sample_idx], n_neighbors=2, return_distance=True)[0][:, 1]
    mins, maxs = X.min(0), X.max(0)
    u_pts = rng.uniform(mins, maxs, size=(m, d))
    u = nn.kneighbors(u_pts, n_neighbors=1, return_distance=True)[0].ravel()
    denom = u.sum() + w.sum()
    return float(u.sum() / denom) if denom else np.nan

def cluster_balance(labels: np.ndarray) -> float:
    """Normalized entropy in [0,1]; higher is more balanced. Noise (-1) ignored."""
    lab = labels[labels >= 0]
    if lab.size == 0:
        return 0.0
    counts = np.bincount(lab)
    k = (counts > 0).sum()
    if k <= 1:
        return 0.0
    p = counts / counts.sum()
    entropy = -(p * np.log(p + 1e-12)).sum()
    return float(entropy / math.log(k))

# -------------------------------------------------------------------
# Configs
# -------------------------------------------------------------------
@dataclass
class PreprocessConfig:
    binary_cols: List[str]
    prob_cols: List[str]
    cont_cols: List[str]
    cat_cols: List[str]
    scaler: str = "robust"  # "robust" | "standard" | "minmax"
    row_norm: str = "l2"    # "l1" | "l2" | None

@dataclass
class ModelSpec:
    name: str
    factory: Callable[[Dict], ClusterMixin]
    param_grid: Dict
    metric: str = "euclidean"  # used where applicable

@dataclass
class TSNEConfig:
    max_points: int = 7000
    perplexity: int = 30
    learning_rate: str | float = "auto"
    random_state: int = 42
    init: str = "pca"
    n_iter: int = 1000

# -------------------------------------------------------------------
# Builders
# -------------------------------------------------------------------
def make_preprocess(cfg: PreprocessConfig) -> Pipeline:
    if cfg.scaler == "robust":
        scaler = RobustScaler()
    elif cfg.scaler == "standard":
        scaler = StandardScaler()
    elif cfg.scaler == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unsupported scaler '{cfg.scaler}'. Choose robust|standard|minmax.")
    ct = ColumnTransformer(
        transformers=[
            ("bin", Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))]), cfg.binary_cols),
            ("cont", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", scaler)]), cfg.cont_cols),
            ("prob", Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))]), cfg.prob_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cfg.cat_cols),
        ],
        remainder="drop",
    )
    steps: List[Tuple[str, object]] = [("preprocess", ct)]
    if cfg.row_norm:
        steps.append(("row_norm", Normalizer(norm=cfg.row_norm)))
    return Pipeline(steps)

# -------------------------------------------------------------------
# Evaluation helper
# -------------------------------------------------------------------
def evaluate_labels(X: np.ndarray, labels: np.ndarray, metric: str) -> Dict[str, float]:
    uniq = np.unique(labels[labels >= 0])
    if uniq.size < 2:
        return {"silhouette": np.nan, "calinski": np.nan, "davies": np.nan, "hopkins": np.nan, "balance": 0.0}
    mask = labels != -1
    X_eval = X[mask]
    y_eval = labels[mask]
    try:
        sil = float(silhouette_score(X_eval, y_eval, metric=metric))
    except Exception:
        sil = np.nan
    try:
        cal = float(calinski_harabasz_score(X_eval, y_eval))
    except Exception:
        cal = np.nan
    try:
        dav = float(davies_bouldin_score(X_eval, y_eval))
    except Exception:
        dav = np.nan
    hop = hopkins_statistic(X_eval)
    bal = cluster_balance(labels)
    return {"silhouette": sil, "calinski": cal, "davies": dav, "hopkins": hop, "balance": bal}

def rank_and_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank and score clustering results using normalized metrics.
    
    Normalization approach:
    - Silhouette: [-1, 1] → [0, 1] via (x + 1) / 2
    - Calinski-Harabasz: [0, ∞) → [0, 1] via min-max scaling
    - Davies-Bouldin: [0, ∞) → [0, 1] via 1 - min-max (inverted, lower is better)
    - Hopkins: [0, 1] → [0, 1] (already normalized)
    - Balance: [0, 1] → [0, 1] (already normalized)
    
    Weights (sum to 1.0):
    - Silhouette: 0.60 (primary cluster quality)
    - Hopkins: 0.15 (clusterability)
    - Calinski: 0.10 (separation)
    - Balance: 0.10 (cluster size distribution)
    - Davies: 0.05 (compactness)
    """
    df = df.copy()
    
    # Normalize silhouette from [-1, 1] to [0, 1]
    df["norm_sil"] = (df["silhouette"] + 1) / 2
    
    # Min-max normalize Calinski (higher is better)
    cal_min, cal_max = df["calinski"].min(), df["calinski"].max()
    if cal_max > cal_min:
        df["norm_cal"] = (df["calinski"] - cal_min) / (cal_max - cal_min)
    else:
        df["norm_cal"] = 0.5
    
    # Invert and normalize Davies-Bouldin (lower is better)
    dav_min, dav_max = df["davies"].min(), df["davies"].max()
    if dav_max > dav_min:
        df["norm_dav"] = 1 - (df["davies"] - dav_min) / (dav_max - dav_min)
    else:
        df["norm_dav"] = 0.5
    
    # Hopkins and balance are already in [0, 1]
    df["norm_hop"] = df["hopkins"]
    df["norm_bal"] = df["balance"]
    
    # Replace NaN with 0 (penalize failed metrics)
    df[["norm_sil", "norm_cal", "norm_dav", "norm_hop", "norm_bal"]] = (
        df[["norm_sil", "norm_cal", "norm_dav", "norm_hop", "norm_bal"]].fillna(0)
    )
    
    # Weighted score (heavy on silhouette)
    df["score"] = (
        1 * df["norm_sil"]
        + 0.10 * df["norm_hop"]
        + 0.10 * df["norm_cal"]
        + 0.10 * df["norm_bal"]
        + 0.10 * df["norm_dav"]
    )
    
    # Drop intermediate normalized columns
    return df.drop(
        columns=["norm_sil", "norm_cal", "norm_dav", "norm_hop", "norm_bal"]
    ).sort_values("score", ascending=False)

# -------------------------------------------------------------------
# Main experiment runner
# -------------------------------------------------------------------
def run_clustering(
    df: pd.DataFrame,
    preprocess_cfg: PreprocessConfig,
    model_specs: Iterable[ModelSpec],
    sample_n: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    data = df if sample_n is None or sample_n >= len(df) else df.sample(sample_n, random_state=42)
    pipe = make_preprocess(preprocess_cfg)
    X_emb = pipe.fit_transform(data)
    if hasattr(X_emb, "toarray"):
        X_emb = X_emb.toarray()

    results: List[Dict] = []
    
    # Calculate total number of parameter combinations
    total_combos = sum(len(list(ParameterGrid(spec.param_grid))) for spec in model_specs)
    
    # Create progress bar if verbose
    pbar = tqdm(total=total_combos, desc="Grid Search", disable=not verbose)

    for spec in model_specs:
        for params in ParameterGrid(spec.param_grid):
            # Skip invalid ward + non-euclidean combinations
            if params.get("linkage") == "ward" and params.get("metric") != "euclidean":
                pbar.update(1)
                continue
            
            merged_params = dict(params)
            if "metric" not in merged_params:
                merged_params["metric"] = spec.metric
            model = spec.factory(merged_params)
            labels = model.fit_predict(X_emb)

            metrics = evaluate_labels(X_emb, labels, metric=merged_params.get("metric", spec.metric))
            noise_frac = float((labels == -1).mean())
            n_clusters = int(len(set(labels) - {-1}))

            row = {
                "model": spec.name,
                "params": params,
                "scaler": preprocess_cfg.scaler,
                "row_norm": preprocess_cfg.row_norm,
                "n_clusters": n_clusters,
                "noise_frac": noise_frac,
                **metrics,
            }
            results.append(row)
            
            # Update progress bar with current best score
            if verbose and results:
                current_best = max(results, key=lambda r: r.get("silhouette", -1))
                pbar.set_postfix({
                    "model": spec.name,
                    "clusters": n_clusters,
                    "sil": f"{metrics['silhouette']:.3f}" if not np.isnan(metrics['silhouette']) else "N/A"
                })
            pbar.update(1)
    
    pbar.close()

    results_df = rank_and_score(pd.DataFrame(results))
    if results_df.empty:
        return results_df, {"best": {}, "embedding": X_emb, "pipeline": pipe}

    # Select best row and rebuild best model/labels safely (works for HDBSCAN too)
    best_row = results_df.iloc[0].to_dict()
    best_name = best_row["model"]
    best_params = best_row["params"]
    spec_map = {spec.name: spec for spec in model_specs}
    best_spec = spec_map[best_name]
    
    # Merge params properly for best model
    merged_best_params = dict(best_params)
    if "metric" not in merged_best_params:
        merged_best_params["metric"] = best_spec.metric
    
    best_model = best_spec.factory(merged_best_params)
    best_labels = best_model.fit_predict(X_emb)

    artifacts = {
        "best": best_row,
        "best_model_name": best_name,
        "best_params": best_params,
        "best_labels": best_labels,
        "embedding": X_emb,
        "pipeline": pipe,
    }
    return results_df, artifacts

# -------------------------------------------------------------------
# t-SNE 3D plotting
# -------------------------------------------------------------------
def plot_tsne_3d(
    X_emb: np.ndarray,
    labels: np.ndarray,
    title: str,
    tsne_cfg: TSNEConfig,
    hover_df: Optional[pd.DataFrame] = None,
) -> None:
    n = X_emb.shape[0]
    rng = np.random.default_rng(tsne_cfg.random_state)
    take = min(tsne_cfg.max_points, n)
    idx = rng.choice(n, size=take, replace=False) if take < n else np.arange(n)

    perp = max(5, min(tsne_cfg.perplexity, take - 1))
    tsne_kwargs = dict(
        n_components=3,
        perplexity=perp,
        learning_rate=tsne_cfg.learning_rate,
        init=tsne_cfg.init,
        random_state=tsne_cfg.random_state,
    )
    if "n_iter" in TSNE.__init__.__code__.co_varnames:
        tsne_kwargs["n_iter"] = tsne_cfg.n_iter

    tsne = TSNE(**tsne_kwargs)
    coords = tsne.fit_transform(X_emb[idx])
    df_plot = pd.DataFrame(coords, columns=["tsne1", "tsne2", "tsne3"])
    df_plot["cluster"] = np.where(labels[idx] == -1, "noise", labels[idx].astype(str))

    if hover_df is not None:
        common_cols = [c for c in hover_df.columns if c not in df_plot.columns]
        df_plot = pd.concat(
            [df_plot.reset_index(drop=True), hover_df.iloc[idx][common_cols].reset_index(drop=True)],
            axis=1,
        )

    fig = px.scatter_3d(
        df_plot,
        x="tsne1",
        y="tsne2",
        z="tsne3",
        color="cluster",
        opacity=0.75,
        title=title + f" (n={take}, perplexity={perp})",
    )
    fig.update_traces(marker=dict(size=3))
    fig.update_layout(legend_title_text="cluster")
    fig.show()

def plot_pca_3d(
    X_emb: np.ndarray,
    labels: np.ndarray,
    title: str,
    max_points: int = 7000,
    random_state: int = 42,
    hover_df: Optional[pd.DataFrame] = None,
) -> None:
    n = X_emb.shape[0]
    rng = np.random.default_rng(random_state)
    take = min(max_points, n)
    idx = rng.choice(n, size=take, replace=False) if take < n else np.arange(n)

    X = X_emb[idx]
    pca = PCA(n_components=3, random_state=random_state)
    coords = pca.fit_transform(X)

    df_plot = pd.DataFrame(coords, columns=["pc1", "pc2", "pc3"])
    df_plot["cluster"] = np.where(labels[idx] == -1, "noise", labels[idx].astype(str))

    if hover_df is not None:
        common_cols = [c for c in hover_df.columns if c not in df_plot.columns]
        df_plot = pd.concat(
            [df_plot.reset_index(drop=True), hover_df.iloc[idx][common_cols].reset_index(drop=True)],
            axis=1,
        )

    explained = pca.explained_variance_ratio_
    fig = px.scatter_3d(
        df_plot,
        x="pc1",
        y="pc2",
        z="pc3",
        color="cluster",
        opacity=0.75,
        title=(
            title
            + f" (n={take}, explained={explained[0]:.2%}/{explained[1]:.2%}/{explained[2]:.2%})"
        ),
    )
    fig.update_traces(marker=dict(size=3))
    fig.update_layout(legend_title_text="cluster")
    fig.show()