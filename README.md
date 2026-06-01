# Public Policy Explainability with SHAP

Published: yes
Medium: [https://medium.com/@kyle-t-jones/explainability-in-public-policy-analysis-5135e4dc1943](https://medium.com/@kyle-t-jones/explainability-in-public-policy-analysis-5135e4dc1943)


This project demonstrates model explainability using SHAP (SHapley Additive exPlanations) for public policy analysis.

## Business context

Public policy decisions increasingly rely on complex machine learning models, from predicting economic trends to evaluating social programs and assessing regulatory impacts. These models provide powerful predictive capabilities but often operate as "black boxes," making it difficult to understand how they generate predictions. This lack of transparency undermines trust, accountability, and ethical decision-making, particularly when public welfare is at stake.

Explainable Artificial Intelligence addresses this challenge by making machine learning models transparent and interpretable. Explainability techniques explain how models work, why they make specific predictions, and which factors influence outcomes the most. This enhances trust, improves accountability, and supports ethical decision-making, enabling policymakers to make informed, data-driven choices.

This post explores the theoretical foundations of explainability and how to implement explainability tools in python for public policy analysis.

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # SHAP analysis functions
│   └── plotting.py    # Tufte-style plotting utilities
├── tests/             # Unit tests
├── data/              # Data files
└── images/            # Generated plots and figures
```

## Configuration

Edit `config.yaml` to customize:
- Data source and target column
- Model parameters (test size, n_estimators)
- SHAP visualization options
- Output settings

## SHAP Analysis

SHAP (SHapley Additive exPlanations) provides:
- Feature Importance: Which features matter most
- Feature Effects: How each feature affects predictions
- Individual Explanations: Why specific predictions were made

## Caveats

- Requires a CSV file with a 'Disease_Risk' column (or configure target column in config).
- SHAP computations can be slow for large datasets.
- Model uses Random Forest by default; can be extended to other models.

## Disclaimer

Educational/demo code only. Not financial, safety, or engineering advice. Use at your own risk. Verify results independently before any production or operational use.

## License

MIT — see [LICENSE](LICENSE).