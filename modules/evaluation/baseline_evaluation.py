"""
Random baseline product evaluation pipeline.
Tests products on random customer samples without clustering intelligence.
"""

import logging
from typing import Dict, Tuple, List, Any
import pandas as pd
from pathlib import Path
import random

from ..agent.client import prepare_llm_payload, batch_process_customers
from .cluster_evaluation import EvaluationMetrics

# Configure logger
logger = logging.getLogger(__name__)


class BaselineEvaluator:
    """Evaluates product performance on random customer samples."""
    
    def __init__(
        self,
        product_catalog: Dict[str, Any],
        api_key: str,
        n_iterations: int = 3,
        sample_size: int = 30
    ):
        """
        Initialize baseline evaluator.
        
        Args:
            product_catalog: Product catalog (dict or Product dataclass instances)
            api_key: API key for LLM service
            n_iterations: Number of evaluation iterations per product
            sample_size: Number of random customers per iteration
        """
        self.product_catalog = product_catalog
        self.api_key = api_key
        self.n_iterations = n_iterations
        self.sample_size = sample_size
    
    def _get_product_attribute(self, product: Any, attr: str) -> Any:
        """Safely extract attribute from product (works with dict or dataclass)."""
        return getattr(product, attr) if hasattr(product, attr) else product.get(attr)
    
    def _evaluate_iteration(
        self,
        sample: pd.DataFrame,
        product_key: str,
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
                api_key=self.api_key,
                use_system_prompt=True
            )
            
            # Validate results
            if results_df.empty or 'satisfaction_score' not in results_df.columns:
                logger.warning(f"Product {product_key}, iteration {iteration}: No valid results")
                return None
            
            # Calculate metrics (decision is now 1/0)
            acceptance_rate = (results_df['decision'] == 1).sum() / len(results_df)
            avg_satisfaction = results_df['satisfaction_score'].mean()
            avg_profile_fit = results_df.get('profile_fit_score', pd.Series([0])).mean()
            
            logger.info(
                f"Product {product_key}, iteration {iteration}/{self.n_iterations}: "
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
            logger.error(f"Product {product_key}, iteration {iteration}: {str(e)}")
            return None
    
    def _evaluate_product(
        self,
        product_key: str,
        customer_df: pd.DataFrame
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Evaluate single product across multiple random samples.
        
        Returns:
            Tuple of (product_summary, iteration_results)
        """
        product = self.product_catalog[product_key]
        product_name = self._get_product_attribute(product, 'name')
        
        logger.info(f"Evaluating product: {product_name}")
        
        iteration_results = []
        metrics = EvaluationMetrics(acceptance_rates=[], satisfaction_scores=[])
        profile_fits = []
        
        # Run iterations with different random samples
        for iteration in range(self.n_iterations):
            # Generate unique random seed per product and iteration
            random_seed = random.randint(1, 1000) + iteration + abs(hash(product_key)) % 1000
            
            sample = customer_df.sample(
                n=self.sample_size,
                random_state=random_seed,
                replace=False
            )
            
            result = self._evaluate_iteration(sample, product_key, iteration + 1)
            
            if result:
                metrics.acceptance_rates.append(result['acceptance_rate'])
                metrics.satisfaction_scores.append(result['avg_satisfaction'])
                profile_fits.append(result['avg_profile_fit_score'])
                
                iteration_results.append({
                    'product_key': product_key,
                    'product_name': product_name,
                    'iteration': iteration + 1,
                    **result
                })
        
        # Aggregate product summary
        if not iteration_results:
            logger.warning(f"Product {product_key}: No successful iterations")
            return None, []
        
        summary = {
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
            f"Product {product_name} summary: "
            f"Acceptance={summary['acceptance_rate_mean']:.2%} "
            f"(±{summary['acceptance_rate_std']:.2%}), "
            f"Satisfaction={summary['satisfaction_mean']:.2f} "
            f"(±{summary['satisfaction_std']:.2f}), "
        )
        
        return summary, iteration_results
    
    def evaluate(self, customer_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run complete baseline evaluation pipeline.
        
        Args:
            customer_df: DataFrame with customer data
            
        Returns:
            Tuple of (summary_df, raw_results_df):
                - summary_df: Aggregated metrics per product
                - raw_results_df: Individual iteration results
        """
        products = list(self.product_catalog.keys())
        logger.info(
            f"Starting baseline evaluation: {len(products)} products, "
            f"{self.n_iterations} iterations, sample size {self.sample_size}"
        )
        
        all_summaries = []
        all_iterations = []
        
        for product_key in products:
            try:
                summary, iterations = self._evaluate_product(product_key, customer_df)
                
                if summary:
                    all_summaries.append(summary)
                    all_iterations.extend(iterations)
                    
            except Exception as e:
                logger.error(f"Product {product_key}: Unexpected error - {str(e)}")
                continue
        
        # Create result DataFrames
        summary_df = pd.DataFrame(all_summaries)
        raw_df = pd.DataFrame(all_iterations)
        
        logger.info(
            f"Baseline complete: {len(summary_df)} products evaluated, "
            f"{len(raw_df)} total iterations"
        )
        
        return summary_df, raw_df
    
    def save_results(
        self,
        summary_df: pd.DataFrame,
        raw_df: pd.DataFrame,
        output_dir: str = "../data",
        prefix: str = "baseline_evaluation"
    ) -> None:
        """
        Save evaluation results to parquet files.
        
        Args:
            summary_df: Summary DataFrame to save
            raw_df: Raw results DataFrame to save
            output_dir: Directory to save files (default: "../data")
            prefix: Prefix for output filenames (default: "baseline_evaluation")
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        summary_file = output_path / f"{prefix}_summary.parquet"
        raw_file = output_path / f"{prefix}_raw.parquet"
        
        summary_df.to_parquet(summary_file, index=False)
        raw_df.to_parquet(raw_file, index=False)
        
        logger.info(f"Results saved: {summary_file}, {raw_file}")


# Update the convenience function
def run_random_baseline_pipeline(
    clustered_df: pd.DataFrame,
    product_catalog: Dict[str, Any],
    api_key: str,
    n_iterations: int = 3,
    sample_size_per_product: int = 30,
    save_results: bool = False,
    output_dir: str = "../data/1_agent_evaluation",
    prefix: str = "baseline_evaluation"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function for random baseline evaluation pipeline.
    
    Args:
        clustered_df: DataFrame with customer data (cluster column ignored)
        product_catalog: Product catalog (dict or Product dataclass instances)
        api_key: API key for LLM service
        n_iterations: Number of evaluation iterations per product
        sample_size_per_product: Number of random customers per iteration
        save_results: Whether to save results to disk (default: False)
        output_dir: Directory to save files (default: "../data")
        prefix: Prefix for output filenames (default: "baseline_evaluation")
        
    Returns:
        Tuple of (summary_df, raw_results_df):
            - summary_df: Aggregated metrics per product
            - raw_results_df: Individual iteration results
    """
    evaluator = BaselineEvaluator(
        product_catalog=product_catalog,
        api_key=api_key,
        n_iterations=n_iterations,
        sample_size=sample_size_per_product
    )
    
    summary_df, raw_df = evaluator.evaluate(clustered_df)
    
    if save_results:
        evaluator.save_results(summary_df, raw_df, output_dir, prefix)
    
    return summary_df, raw_df