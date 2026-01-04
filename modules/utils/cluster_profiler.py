"""
Cluster profiling and visualization utilities.
Provides EDA tools for analyzing clustered customer data.
"""

import logging
from typing import List, Optional, Set
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.feature_selection import f_classif

# Configure logger
logger = logging.getLogger(__name__)


class ClusterProfiler:
    """
    Quick EDA utilities for clustered data.
    
    Provides methods for:
    - Mean/z-score profiles
    - Top feature deviations
    - ANOVA F-scores
    - Boxplots, KDEs, radar charts, categorical distributions
    """

    def __init__(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
        cluster_col: str = "cluster",
        exclude: Optional[Set[str]] = None
    ):
        """
        Initialize cluster profiler.
        
        Args:
            df: DataFrame with customer features
            labels: Cluster labels (array-like)
            cluster_col: Name for cluster column (default: "cluster")
            exclude: Set of column names to exclude from analysis
        """
        self.df = df.copy()
        self.df[cluster_col] = labels
        self.cluster_col = cluster_col
        self.exclude = set(exclude or [])
        
        # Identify numeric columns
        num_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        if cluster_col in num_cols:
            num_cols.remove(cluster_col)
        self.numeric = [c for c in num_cols if c not in self.exclude]
        
        # Identify categorical columns
        self.categorical = [
            c for c in self.df.select_dtypes(include=["object", "category"]).columns
            if c not in self.exclude
        ]
        
        logger.info(
            f"Initialized profiler: {len(self.numeric)} numeric, "
            f"{len(self.categorical)} categorical features"
        )

    def mean_profiles(self, zscore: bool = False) -> pd.DataFrame:
        """
        Compute mean feature values per cluster.
        
        Args:
            zscore: If True, return z-score normalized means
            
        Returns:
            DataFrame with cluster means (optionally z-scored)
        """
        grouped = self.df.groupby(self.cluster_col)[self.numeric].mean()
        
        if not zscore:
            return grouped
        
        # Z-score normalization
        overall_mean = self.df[self.numeric].mean()
        overall_std = self.df[self.numeric].std().replace(0, np.nan)
        
        return (grouped - overall_mean) / overall_std

    def top_feature_deviations(self, k: int = 5) -> pd.DataFrame:
        """
        Identify top k features with largest deviation from overall mean per cluster.
        
        Args:
            k: Number of top features to return per cluster
            
        Returns:
            DataFrame with cluster, feature, deviation metrics
        """
        grouped = self.df.groupby(self.cluster_col)[self.numeric].mean()
        overall = self.df[self.numeric].mean()
        diffs = (grouped - overall).abs()
        
        rows = []
        for cluster in diffs.index:
            topk = diffs.loc[cluster].sort_values(ascending=False).head(k)
            for feat, dev in topk.items():
                rows.append({
                    "cluster": cluster,
                    "feature": feat,
                    "mean_deviation": dev,
                    "cluster_mean": grouped.loc[cluster, feat],
                    "overall_mean": overall[feat],
                })
        
        result = pd.DataFrame(rows).sort_values(
            ["cluster", "mean_deviation"],
            ascending=[True, False]
        )
        
        logger.info(f"Computed top {k} deviations for {len(diffs)} clusters")
        return result

    def anova_scores(self) -> pd.DataFrame:
        """
        Compute ANOVA F-scores for all numeric features.
        
        Returns:
            DataFrame with features, F-scores, and p-values sorted by F-score
        """
        X = self.df[self.numeric]
        y = self.df[self.cluster_col]
        
        f_vals, p_vals = f_classif(X, y)
        
        result = pd.DataFrame({
            "Feature": self.numeric,
            "F Score": f_vals,
            "P Value": p_vals
        })
        
        result["F Score"] = result["F Score"].round(2)
        result["P Value"] = result["P Value"].apply(lambda x: f"{x:.2e}")
        
        return result.sort_values("F Score", ascending=False).reset_index(drop=True)

    def boxplots(self, features: Optional[List[str]] = None) -> None:
        """
        Generate boxplots for features across clusters.
        
        Args:
            features: List of features to plot (default: all numeric)
        """
        feats = features or self.numeric
        
        for feat in feats:
            plt.figure(figsize=(6, 4))
            sns.boxplot(x=self.cluster_col, y=feat, data=self.df)
            plt.title(f"{feat} by cluster")
            plt.tight_layout()
            plt.show()
        
        logger.info(f"Generated boxplots for {len(feats)} features")

    def kde_overlays(self, features: Optional[List[str]] = None) -> None:
        """
        Generate overlaid KDE plots for features across clusters.
        
        Args:
            features: List of features to plot (default: all numeric)
        """
        feats = features or self.numeric
        
        for feat in feats:
            plt.figure(figsize=(6, 4))
            for cluster in sorted(self.df[self.cluster_col].unique()):
                cluster_data = self.df.loc[self.df[self.cluster_col] == cluster, feat]
                sns.kdeplot(cluster_data, label=f"c{cluster}", fill=True, alpha=0.25)
            
            plt.title(f"KDE of {feat}")
            plt.legend()
            plt.tight_layout()
            plt.show()
        
        logger.info(f"Generated KDE plots for {len(feats)} features")

    def categorical_bars(self) -> None:
        """Generate stacked bar charts for categorical features across clusters."""
        if not self.categorical:
            logger.warning("No categorical features to plot")
            return
        
        for col in self.categorical:
            plt.figure(figsize=(7, 4))
            
            # Calculate proportions
            prop = (
                self.df.groupby([self.cluster_col, col])
                .size()
                .groupby(level=0)
                .apply(lambda x: 100 * x / x.sum())
                .reset_index(name="pct")
            )
            
            sns.barplot(x=self.cluster_col, y="pct", hue=col, data=prop)
            plt.title(f"{col} distribution (%)")
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()
            plt.show()
        
        logger.info(f"Generated categorical plots for {len(self.categorical)} features")

    def radar(self, top_n: int = 5) -> None:
        """
        Generate radar chart comparing clusters on top N features.
        
        Args:
            top_n: Number of top features by ANOVA F-score to display
        """
        means = self.mean_profiles(zscore=True)
        top_feats = self.anova_scores().head(top_n)["Feature"].tolist()
        data = means[top_feats]

        fig = go.Figure()
        
        for cluster, row in data.iterrows():
            vals = row.tolist()
            vals += vals[:1]  # Close the loop
            
            fig.add_trace(go.Scatterpolar(
                r=vals,
                theta=top_feats + [top_feats[0]],
                fill='toself',
                name=f"cluster {cluster}",
                opacity=0.55
            ))

        fig.update_layout(
            title=f"Z-score radar (top {top_n} features by ANOVA)",
            width=800,
            height=700,
            polar=dict(radialaxis=dict(visible=True, tickfont_size=10)),
            legend=dict(orientation="v", x=1.05, y=1.0)
        )
        
        fig.show()
        logger.info(f"Generated radar chart for top {top_n} features")

    def summary_report(self, k: int = 5) -> None:
        """
        Generate comprehensive summary report.
        
        Args:
            k: Number of top features to show in deviation analysis
        """
        print("=" * 80)
        print("CLUSTER PROFILING REPORT")
        print("=" * 80)
        
        print(f"\nClusters: {sorted(self.df[self.cluster_col].unique())}")
        print(f"Total samples: {len(self.df)}")
        print(f"Numeric features: {len(self.numeric)}")
        print(f"Categorical features: {len(self.categorical)}")
        
        print("\n" + "=" * 80)
        print("CLUSTER SIZE DISTRIBUTION")
        print("=" * 80)
        print(self.df[self.cluster_col].value_counts().sort_index())
        
        print("\n" + "=" * 80)
        print(f"TOP {k} FEATURE DEVIATIONS PER CLUSTER")
        print("=" * 80)
        print(self.top_feature_deviations(k=k))
        
        print("\n" + "=" * 80)
        print("ANOVA F-SCORES (Top 10)")
        print("=" * 80)
        print(self.anova_scores().head(10))