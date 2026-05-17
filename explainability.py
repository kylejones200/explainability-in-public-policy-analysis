import logging

import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    data = pd.read_csv("public_health_data.csv")
    X = data.drop(columns=["Disease_Risk"])
    y = data["Disease_Risk"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)


def prepare_explainer() -> None:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values[1], X_test, feature_names=X.columns)
    shap.force_plot(
        explainer.expected_value[1],
        shap_values[1][0],
        X_test[0],
        feature_names=X.columns,
    )


def configure_logging_2() -> None:
    configure_logging()

    prepare_explainer()


def main() -> None:
    configure_logging_2()


if __name__ == "__main__":
    main()
