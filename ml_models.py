"""
Regression models and diagnostics for spectral perturbation prediction.

Models: Ridge, ElasticNet, Random Forest, XGBoost (optional).
Diagnostics: predicted-vs-true, residual heatmaps, feature importance,
partial dependence.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import PartialDependenceDisplay


# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_ridge(X, y, alpha=1.0, cv=5):
    """Ridge regression with cross-validation."""
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = Ridge(alpha=alpha)
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    r2_scores = cross_val_score(model, Xs, y, cv=kf, scoring="r2")
    neg_rmse = cross_val_score(
        model, Xs, y, cv=kf, scoring="neg_root_mean_squared_error"
    )
    model.fit(Xs, y)
    return {
        "model": model, "scaler": scaler,
        "cv_r2": r2_scores, "cv_rmse": -neg_rmse,
        "mean_r2": r2_scores.mean(), "mean_rmse": (-neg_rmse).mean(),
        "coefficients": model.coef_,
    }


def train_elasticnet(X, y, alpha=0.1, l1_ratio=0.5, cv=5):
    """Elastic Net regression."""
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000)
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    r2_scores = cross_val_score(model, Xs, y, cv=kf, scoring="r2")
    neg_rmse = cross_val_score(
        model, Xs, y, cv=kf, scoring="neg_root_mean_squared_error"
    )
    model.fit(Xs, y)
    return {
        "model": model, "scaler": scaler,
        "cv_r2": r2_scores, "cv_rmse": -neg_rmse,
        "mean_r2": r2_scores.mean(), "mean_rmse": (-neg_rmse).mean(),
        "coefficients": model.coef_,
    }


def train_rf(X, y, n_estimators=300, cv=5):
    """Random Forest regression."""
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = RandomForestRegressor(
        n_estimators=n_estimators, random_state=42, n_jobs=-1, max_depth=8
    )
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    r2_scores = cross_val_score(model, Xs, y, cv=kf, scoring="r2")
    neg_rmse = cross_val_score(
        model, Xs, y, cv=kf, scoring="neg_root_mean_squared_error"
    )
    model.fit(Xs, y)
    return {
        "model": model, "scaler": scaler,
        "cv_r2": r2_scores, "cv_rmse": -neg_rmse,
        "mean_r2": r2_scores.mean(), "mean_rmse": (-neg_rmse).mean(),
        "feature_importances": model.feature_importances_,
    }


def train_xgboost(X, y, cv=5):
    """XGBoost regression. Returns None if not installed."""
    if not HAS_XGBOOST:
        return None

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = xgb.XGBRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        random_state=42, n_jobs=-1,
    )
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    r2_scores = cross_val_score(model, Xs, y, cv=kf, scoring="r2")
    neg_rmse = cross_val_score(
        model, Xs, y, cv=kf, scoring="neg_root_mean_squared_error"
    )
    model.fit(Xs, y)
    return {
        "model": model, "scaler": scaler,
        "cv_r2": r2_scores, "cv_rmse": -neg_rmse,
        "mean_r2": r2_scores.mean(), "mean_rmse": (-neg_rmse).mean(),
        "feature_importances": model.feature_importances_,
    }


# ---------------------------------------------------------------------------
# Predictions and diagnostics
# ---------------------------------------------------------------------------

def predict(model_result, X):
    """Generate predictions from a trained model result."""
    Xs = model_result["scaler"].transform(X)
    return model_result["model"].predict(Xs)


def residuals(y_true, y_pred):
    """Compute residuals."""
    return y_true - y_pred


def model_summary(model_result, name):
    """Print a summary line for a model."""
    print(f"  {name:15s}  R2={model_result['mean_r2']:+.4f}  "
          f"RMSE={model_result['mean_rmse']:.6f}")


def get_feature_importance_df(model_result, feature_names):
    """Build sorted feature importance DataFrame from tree models."""
    if "feature_importances" not in model_result:
        return None
    return pd.DataFrame({
        "feature": feature_names,
        "importance": model_result["feature_importances"],
    }).sort_values("importance", ascending=False).reset_index(drop=True)


def get_coefficient_df(model_result, feature_names):
    """Build sorted coefficient DataFrame from linear models."""
    if "coefficients" not in model_result:
        return None
    return pd.DataFrame({
        "feature": feature_names,
        "coefficient": model_result["coefficients"],
        "abs_coefficient": np.abs(model_result["coefficients"]),
    }).sort_values("abs_coefficient", ascending=False).reset_index(drop=True)
