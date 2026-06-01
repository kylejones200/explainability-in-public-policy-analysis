import logging
from pathlib import Path

from src.core import (
    compute_shap_values,
    load_data,
    prepare_data,
    save_shap_plots,
    train_model,
)


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def run_shap_analysis(data_path: Path, output_dir: Path) -> None:
    X, y = load_data(data_path)
    X_train, X_test, y_train, _y_test, _scaler = prepare_data(X, y)
    model = train_model(X_train, y_train)
    explainer, shap_values = compute_shap_values(model, X_test, X.columns.tolist())
    save_shap_plots(explainer, shap_values, X_test, X.columns.tolist(), output_dir)


def main() -> None:
    configure_logging()
    run_shap_analysis(Path("public_health_data.csv"), Path("."))


if __name__ == "__main__":
    main()
