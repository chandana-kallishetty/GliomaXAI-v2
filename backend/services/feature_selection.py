"""
feature_selection.py
====================
Implements the four-stage dimensionality reduction pipeline:
1. Variance Threshold
2. Correlation Filtering
3. LASSO (L1 penalty) Select
4. Recursive Feature Elimination (RFE)
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor

def run_feature_selection(X: pd.DataFrame, y: np.ndarray, target_features_count: int = 15) -> List[str]:
    """
    Executes the 4-stage feature selection pipeline to reduce dimensionality of extracted radiomics features.
    
    Parameters
    ----------
    X : pd.DataFrame
        DataFrame of extracted features (N_samples x N_features).
    y : np.ndarray
        Regression target (e.g. survival time or status).
    target_features_count : int
        Final number of features to select.
        
    Returns
    -------
    selected_features : List[str]
        List of feature names that survived the selection pipeline.
    """
    if X.empty:
        return []
        
    features = X.columns.tolist()
    
    # --- Stage 1: Variance Thresholding (Remove near-constant features) ---
    try:
        # Lower threshold to keep features with slight variance
        selector = VarianceThreshold(threshold=0.01)
        selector.fit(X)
        stage1_features = X.columns[selector.get_support()].tolist()
        if len(stage1_features) < target_features_count:
            stage1_features = features
    except Exception as e:
        print(f"[feature_selection] Stage 1 (Variance Threshold) skipped: {e}")
        stage1_features = features
        
    X_s1 = X[stage1_features]
    
    # --- Stage 2: Correlation Filtering (Remove highly collinear features) ---
    try:
        corr_matrix = X_s1.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        # Find features with correlation greater than 0.90
        to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]
        stage2_features = [f for f in stage1_features if f not in to_drop]
        if len(stage2_features) < target_features_count:
            stage2_features = stage1_features
    except Exception as e:
        print(f"[feature_selection] Stage 2 (Correlation Filter) skipped: {e}")
        stage2_features = stage1_features
        
    X_s2 = X_s1[stage2_features]
    
    # --- Stage 3: LASSO (Least Absolute Shrinkage and Selection Operator) ---
    try:
        # Standardize target if needed
        lasso = LassoCV(cv=5, max_iter=2000, random_state=42)
        lasso.fit(X_s2, y)
        coef = pd.Series(lasso.coef_, index=X_s2.columns)
        stage3_features = coef[coef != 0].index.tolist()
        if len(stage3_features) < target_features_count:
            # Fallback: keep top coefficients
            sorted_coef = coef.abs().sort_values(ascending=False)
            stage3_features = sorted_coef.head(max(target_features_count, len(stage3_features))).index.tolist()
    except Exception as e:
        print(f"[feature_selection] Stage 3 (LASSO) skipped: {e}")
        stage3_features = stage2_features
        
    X_s3 = X_s2[stage3_features]
    
    # --- Stage 4: Recursive Feature Elimination (RFE) ---
    try:
        estimator = RandomForestRegressor(n_estimators=50, random_state=42)
        n_features_to_select = min(target_features_count, len(stage3_features))
        rfe = RFE(estimator=estimator, n_features_to_select=n_features_to_select, step=1)
        rfe.fit(X_s3, y)
        selected_features = X_s3.columns[rfe.get_support()].tolist()
    except Exception as e:
        print(f"[feature_selection] Stage 4 (RFE) skipped: {e}")
        selected_features = stage3_features[:target_features_count]
        
    return selected_features
