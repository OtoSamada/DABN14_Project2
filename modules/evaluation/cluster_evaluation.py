"""
Cluster-based product evaluation pipeline.
Evaluates how well products perform when targeted to specific customer clusters.
"""

import logging
from typing import Dict, Tuple, List, Any
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
import random

from ..agent.client import prepare_llm_payload, batch_process_customers

# Configure logger
logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics with statistical aggregation."""
    
    acceptance_rates: List[float]
    satisfaction_scores: List[float]
    
    @property
    def acceptance_mean(self) -> float:
        return pd.Series(self.acceptance_rates).mean()
    
    @property
    def acceptance_std(self) -> float:
        return pd.Series(self.acceptance_rates).std()
    
    @property
    def satisfaction_mean(self) -> float:
        return pd.Series(self.satisfaction_scores).mean()
    
    @property
    def satisfaction_std(self) -> float:
        return pd.Series(self.satisfaction_scores).std()


class ClusterEvaluator:
    """Evaluates product performance on targeted customer clusters."""
    
    def __init__(
        self,
        product_catalog: Dict[str, Any],
        api_key: str,
        n_iterations: int = 5,
        sample_size: int = 50
    ):
        """
        Initialize cluster evaluator.
        
        Args:
            product_catalog: Product catalog (dict or Product dataclass instances)
            api_key: API key for LLM service
            n_iterations: Number of evaluation iterations per cluster
            sample_size: Number of customers to sample per iteration
        """
        self.product_catalog = product_catalog
        self.api_key = api_key
        self.n_iterations = n_iterations
        self.sample_size = sample_size
    
    def _get_product_attribute(self, product: Any, attr: str) -> Any:
        """Safely extract attribute from product (works with dict or dataclass)."""
        return getattr(product, attr) if hasattr(product, attr) else product.get(attr)
    
    def _find_matching_product(self, cluster_id: int) -> Tuple[str, str]:
        """
        Find first product targeting the given cluster.
        
        Returns:
            Tuple of (product_key, product_name)
        Raises:
            ValueError: If no matching product found
        """
        for product_key, product in self.product_catalog.items():
            target_clusters = self._get_product_attribute(product, 'target_cluster')
            if cluster_id in target_clusters:
                product_name = self._get_product_attribute(product, 'name')
                return product_key, product_name
        
        raise ValueError(f"No product found targeting cluster {cluster_id}")
    
    def _evaluate_iteration(
        self,
        sample: pd.DataFrame,
        product_key: str,
        cluster_id: int,
        iteration: int
    ) -> Dict[str, Any]:
        """
        Run single evaluation iteration.
        
        Returns:
            Dictionary with iteration metrics or None if failed
        """
        try:
            # Prepare customer payloads
            payloads = [prepare_llm_payload(row) for _, row in sample.iterrows()]
            
            # Run LLM evaluation
            results_df = batch_process_customers(
                customers=payloads,
                product_key=product_key,
                product_catalog=self.product_catalog,
                api_key=self.api_key
            )
            
            # Validate results
            if results_df.empty or 'satisfaction_score' not in results_df.columns:
                logger.warning(f"Cluster {cluster_id}, iteration {iteration}: No valid results")
                return None
            
            # Calculate metrics (decision is now 1/0)
            acceptance_rate = (results_df['decision'] == 1).sum() / len(results_df)
            avg_satisfaction = results_df['satisfaction_score'].mean()
            avg_profile_fit = results_df.get('profile_fit_score', pd.Series([0])).mean()
            
            logger.info(
                f"Cluster {cluster_id}, iteration {iteration}/{self.n_iterations}: "
                f"Accept={acceptance_rate:.2%}, Satisfaction={avg_satisfaction:.2f}, "
                f"ProfileFit={avg_profile_fit:.2f}"
            )
            
            return {
                'acceptance_rate': acceptance_rate,
                'avg_satisfaction': avg_satisfaction,
                'avg_profile_fit_score': avg_profile_fit,
                'sample_size': len(results_df)
            }
            
        except Exception as e:
            logger.error(f"Cluster {cluster_id}, iteration {iteration}: {str(e)}")
            return None

    def _evaluate_cluster(
        self,
        cluster_id: int,
        cluster_data: pd.DataFrame,
        product_key: str,
        product_name: str
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Evaluate single cluster across multiple iterations.
        
        Returns:
            Tuple of (cluster_summary, iteration_results)
        """
        logger.info(f"Evaluating Cluster {cluster_id} with product: {product_name}")
        
        # Adjust sample size if insufficient data
        adjusted_size = min(self.sample_size, len(cluster_data))
        if adjusted_size < self.sample_size:
            logger.warning(
                f"Cluster {cluster_id}: Only {len(cluster_data)} customers available. "
                f"Adjusting sample size to {adjusted_size}"
            )
        
        iteration_results = []
        metrics = EvaluationMetrics(acceptance_rates=[], satisfaction_scores=[])
        
        # Track new metrics
        profile_fits = []
        
        # Run iterations
        for iteration in range(self.n_iterations):
            sample = cluster_data.sample(
                n=adjusted_size,
                random_state=random.randint(1, 1000) + iteration,
                replace=False
            )
            
            result = self._evaluate_iteration(sample, product_key, cluster_id, iteration + 1)
            
            if result:
                metrics.acceptance_rates.append(result['acceptance_rate'])
                metrics.satisfaction_scores.append(result['avg_satisfaction'])
                profile_fits.append(result['avg_profile_fit_score'])
                
                iteration_results.append({
                    'cluster': cluster_id,
                    'product_key': product_key,
                    'product_name': product_name,
                    'iteration': iteration + 1,
                    **result
                })
        
        # Aggregate cluster summary
        if not iteration_results:
            logger.warning(f"Cluster {cluster_id}: No successful iterations")
            return None, []
        
        summary = {
            'cluster': cluster_id,
            'product_key': product_key,
            'product_name': product_name,
            'n_iterations': len(iteration_results),
            'total_customers_tested': sum(r['sample_size'] for r in iteration_results),
            'acceptance_rate_mean': metrics.acceptance_mean,
            'acceptance_rate_std': metrics.acceptance_std,
            'acceptance_rate_min': min(metrics.acceptance_rates),
            'acceptance_rate_max': max(metrics.acceptance_rates),
            'satisfaction_mean': metrics.satisfaction_mean,
            'satisfaction_std': metrics.satisfaction_std,
            'satisfaction_min': min(metrics.satisfaction_scores),
            'satisfaction_max': max(metrics.satisfaction_scores),
            'profile_fit_score_mean': sum(profile_fits) / len(profile_fits),
            'profile_fit_score_std': pd.Series(profile_fits).std(),
        }
        
        logger.info(
            f"Cluster {cluster_id} summary: "
            f"Acceptance={summary['acceptance_rate_mean']:.2%} "
            f"(±{summary['acceptance_rate_std']:.2%}), "
            f"Satisfaction={summary['satisfaction_mean']:.2f} "
            f"(±{summary['satisfaction_std']:.2f}), "
        )
        
        return summary, iteration_results
    
    def evaluate(self, clustered_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run complete cluster evaluation pipeline.
        
        Args:
            clustered_df: DataFrame with customer data and cluster assignments
            
        Returns:
            Tuple of (summary_df, raw_results_df):
                - summary_df: Aggregated metrics per cluster
                - raw_results_df: Individual iteration results
        """
        clusters = sorted(clustered_df['cluster'].unique())
        logger.info(
            f"Starting cluster evaluation: {len(clusters)} clusters, "
            f"{self.n_iterations} iterations, sample size {self.sample_size}"
        )
        
        all_summaries = []
        all_iterations = []
        
        for cluster_id in clusters:
            try:
                # Find matching product
                product_key, product_name = self._find_matching_product(cluster_id)
                
                # Get cluster data
                cluster_data = clustered_df[clustered_df['cluster'] == cluster_id]
                
                # Evaluate cluster
                summary, iterations = self._evaluate_cluster(
                    cluster_id, cluster_data, product_key, product_name
                )
                
                if summary:
                    all_summaries.append(summary)
                    all_iterations.extend(iterations)
                    
            except ValueError as e:
                logger.warning(f"Cluster {cluster_id}: {str(e)}")
                continue
            except Exception as e:
                logger.error(f"Cluster {cluster_id}: Unexpected error - {str(e)}")
                continue
        
        # Create result DataFrames
        summary_df = pd.DataFrame(all_summaries)
        raw_df = pd.DataFrame(all_iterations)
        
        logger.info(
            f"Evaluation complete: {len(summary_df)} clusters evaluated, "
            f"{len(raw_df)} total iterations"
        )
        
        return summary_df, raw_df
    
    def save_results(
        self,
        summary_df: pd.DataFrame,
        raw_df: pd.DataFrame,
        output_dir: str = "../data",
        prefix: str = "cluster_evaluation"
    ) -> None:
        """
        Save evaluation results to parquet files.
        
        Args:
            summary_df: Summary DataFrame to save
            raw_df: Raw results DataFrame to save
            output_dir: Directory to save files (default: "../data")
            prefix: Prefix for output filenames (default: "cluster_evaluation")
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        summary_file = output_path / f"{prefix}_summary.parquet"
        raw_file = output_path / f"{prefix}_raw.parquet"
        
        summary_df.to_parquet(summary_file, index=False)
        raw_df.to_parquet(raw_file, index=False)
        
        logger.info(f"Results saved: {summary_file}, {raw_file}")


def run_cluster_evaluation_pipeline(
    clustered_df: pd.DataFrame,
    product_catalog: Dict[str, Any],
    api_key: str,
    n_iterations: int = 5,
    sample_size: int = 50,
    save_results: bool = False,
    output_dir: str = "../data/1_agent_evaluation",
    prefix: str = "cluster_evaluation"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function for cluster evaluation pipeline.
    
    Args:
        clustered_df: DataFrame with customer data and cluster assignments
        product_catalog: Product catalog (dict or Product dataclass instances)
        api_key: API key for LLM service
        n_iterations: Number of evaluation iterations per cluster
        sample_size: Number of customers to sample per iteration
        save_results: Whether to save results to disk (default: False)
        output_dir: Directory to save files (default: "../data")
        prefix: Prefix for output filenames (default: "cluster_evaluation")
        
    Returns:
        Tuple of (summary_df, raw_results_df):
            - summary_df: Aggregated metrics per cluster
            - raw_results_df: Individual iteration results
    """
    evaluator = ClusterEvaluator(
        product_catalog=product_catalog,
        api_key=api_key,
        n_iterations=n_iterations,
        sample_size=sample_size
    )
    
    summary_df, raw_df = evaluator.evaluate(clustered_df)
    
    if save_results:
        evaluator.save_results(summary_df, raw_df, output_dir, prefix)
    
    return summary_df, raw_df