# Bank Product Recommendation System using Clustering

An end-to-end machine learning project that segments bank customers using unsupervised learning and recommends personalized financial products. The system combines traditional clustering algorithms with LLM-based evaluation to validate recommendation quality.

## Project Overview

This project implements a clustering-based recommendation engine for retail banking products. Customer financial profiles are segmented into distinct personas using Agglomerative Hierarchical Clustering, with each cluster mapped to appropriate banking products (refinance loans, credit lines, savings accounts, etc.). The system is evaluated using both statistical metrics and an LLM agent that simulates customer acceptance behavior.

**Key Technologies:** Python, scikit-learn, HDBSCAN, OpenAI API, feature engineering, representation learning, dimensionality reduction (PCA, t-SNE)

## Repository Structure

```
.
├── data/
│   ├── 0_clustering_data/          # Preprocessed customer features
│   ├── 1_llm_evaluations/          # LLM agent evaluation results
│   ├── bank-additional/            # UCI Bank Marketing dataset
│   └── raw_data/                   # Original Personal Finance ML dataset
│
├── models/                         # Trained clustering models (.joblib)
│   ├── best_hdbscan_model.joblib
│   ├── best_preprocessing_pipeline.joblib
│   └── ...
│
├── modules/                        # Core Python modules
│   ├── agent/                      # LLM-based evaluation agent
│   ├── evaluation/                 # Clustering metrics & validation
│   ├── model_training/             # Preprocessing & clustering pipeline
│   └── utils/                      # Helper functions
│
├── notebooks/                       # Jupyter notebooks
│   ├── model_training_small_fs.ipynb  # Contains clustering model 
│   └── agent_test.ipynb               # these messy files contain LLM evaluation experiments
│
├── product_catalog/                 # Banking product definitions
│   └── product_catalog.py
│
└── Group 11 - Project 2 Report.ipynb  # Complete project report
```

## Usage

The complete analysis workflow is documented in `Group 11 - Project 2 Report.ipynb`. This notebook includes:
- Data preprocessing and feature engineering
- Clustering model training and hyperparameter tuning
- Cluster interpretation and product-persona matching
- LLM-based evaluation framework
- Results and conclusions

To reproduce the analysis, run the cells sequentially from top to bottom.

## Key Results

- **Final Model:** Agglomerative Clustering with 8 clusters (cosine similarity, average linkage)
- **Cluster Quality:** Silhouette Score = 0.73, Calinski-Harabasz = 15,021
- **Recommendation Performance:** 95.6% LLM-simulated acceptance rate vs. 68% for random allocation

For detailed results, methodology, and visualizations, see the full report notebook.

## Authors

Otari Samadashvili & Nana Jaoshvili
Advanced Machine Learning (DABN14) – January 2026