# Assignment 3 - Machine Learning

## Contents

- `linear_reg.py`: Linear regression with RMSE calculation and polynomial pretty-printing.
- `kmeans_clustering.py`: K-means clustering implementation and evaluation.
- `mlp_classifier.py`: Multi-layer perceptron (MLP) classifier using scikit-learn, with basic and advanced hyperparameter tuning.
- `gridworld_policy_iteration.py`: Policy iteration for a gridworld environment, including policy evaluation and improvement.
- `data_process.py`: Utility functions for loading and splitting datasets.
- `GasProperties.csv`: Example dataset for regression and analysis.
- Test files: Unit tests for each main algorithm implementation.

## How to Run

1. **Install dependencies** (if not already installed):
   ```bash
   pip install numpy pandas matplotlib scikit-learn torch
   ```
2. **Run scripts**:
   - For linear regression:
     ```bash
     python linear_reg.py
     ```
   - For MLP classifier:
     ```bash
     python mlp_classifier.py
     ```
   - For gridworld policy iteration:
     ```bash
     python gridworld_policy_iteration.py
     ```
   - For k-means clustering:
     ```bash
     python kmeans_clustering.py
     ```
3. **Run tests**:
   ```bash
   python test_linear_reg.py
   python test_kmeans_clustering.py
   python test_mlp_classifier.py
   python test_gridworld_policy_iteration.py
   ```
