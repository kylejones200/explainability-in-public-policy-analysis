"""Core functions for public policy explainability using SHAP."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")


def load_data(data_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load and prepare data for modeling."""
    data = pd.read_csv(data_path)
    X = data.drop(columns=["Disease_Risk"])
    y = data["Disease_Risk"]
    return X, y


def prepare_data(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.3, random_state: int = 42
) -> tuple:
    """Split and scale data."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, scaler


def train_model(
    X_train: np.ndarray,
    y_train: pd.Series,
    n_estimators: int = 100,
    random_state: int = 42,
):
    """Train Random Forest classifier."""
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def compute_shap_values(model, X_test: np.ndarray, feature_names: list):
    """Compute SHAP values for model explainability."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    return explainer, shap_values


def save_shap_plots(
    explainer, shap_values, X_test: np.ndarray, feature_names: list, output_dir: Path
):
    """Save SHAP visualization plots."""
    shap.summary_plot(shap_values[1], X_test, feature_names=feature_names, show=False)
    plt.savefig(
        output_dir / "shap_summary_plot.png",
        dpi=100,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()

    plt.savefig(
        output_dir / "shap_summary_bar.png",
        dpi=100,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()
