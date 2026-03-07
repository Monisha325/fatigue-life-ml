# Fatigue Life Prediction of Welded Joints in Bridge Structures
### Machine Learning for Structural Reliability Assessment

**A data-driven ML framework for predicting fatigue life of welded bridge joints, trained on 45,315 real experimental observations extracted from 1,666 peer-reviewed publications.(from figshare)**

---

## Motivation

Fatigue failure is one of the leading causes of structural deterioration in welded bridge components. Traditional S-N curve methods cannot fully capture the **nonlinear interactions** between material properties, weld geometry, residual stresses, and variable-amplitude loading histories.

This project develops a complete machine learning pipeline — from raw experimental data to probabilistic fatigue life prediction — to support more accurate, data-informed structural reliability assessment and bridge maintenance strategies.

---

## Dataset

| Property | Details |
|----------|---------|
| **Source** | Deng, C. et al. *Scientific Data, Nature* (2025) |
| **DOI** | https://doi.org/10.6084/m9.figshare.29254265.v2 |
| **Size** | 45,315 stress-life entries |
| **Origin** | Extracted from 1,666 peer-reviewed publications via NLP, image recognition and table parsing |
| **Features** | Stress range, residual stress, weld geometry, material properties, load ratio, welding process parameters |
| **Target** | `log_fatigue_life` — log₁₀ of fatigue life in cycles to failure |

---

## Project Structure

```
Fatigue-Life-ML/
├── data/
│   ├── raw/
│   │   └── Weld-Fatigue-Database-json/
│   │       ├── parameter.json
│   │       └── S-N.json
│   └── processed/
│       └── fatigue_dataset_clean.csv
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_eda_analysis.py
│   ├── 03_model_training.py
│   └── 04_uncertainty_analysis.py
├── models/
│   ├── linear_regression.pkl
│   ├── random_forest.pkl
│   └── xgboost_model.pkl
├── results/
│   ├── figures/
│   └── model_metrics.csv
├── requirements.txt
└── README.md
```

---

## Pipeline

Run the notebooks in the following order:

### Step 1 — Data Preprocessing
**`01_data_preprocessing.ipynb`**

- Loads and parses `parameter.json` and `S-N.json` using `pd.json_normalize()`
- Merges datasets on `dataset_id`
- Engineers domain-informed features: `stress_amplitude`, `mean_stress`, `normalized_stress`, `log_fatigue_life`
- Saves clean dataset to `data/processed/fatigue_dataset_clean.csv`

### Step 2 — Exploratory Data Analysis
**`02_eda_analysis.py`**

| Figure | Description |
|--------|-------------|
| Fatigue Life Distribution | Confirms log-normal behaviour — justifies log-scale target |
| S-N Curve | Validates inverse stress-life relationship against domain theory |
| Correlation Heatmap | Identifies multicollinearity among stress-related features |
| Welding Method vs Fatigue Life | Reveals process-dependent performance variability |
| Residual Stress vs Fatigue Life | Captures influence of thermally-induced stresses |
| Outlier Detection | Retains extreme values consistent with fatigue scatter |
| Stress Feature Distributions | Illustrates wide loading range represented in dataset |

All figures saved to `results/figures/`.

### Step 3 — Model Training and Comparison
**`03_model_training.py`**

Three regression models are trained and evaluated:

| Model | Type | Role |
|-------|------|------|
| Linear Regression | Linear | Baseline — establishes minimum benchmark |
| Random Forest | Nonlinear ensemble | Captures complex feature interactions |
| XGBoost | Gradient boosting | High-performance nonlinear modelling |

- Evaluation metrics: RMSE, MAE, R²
- Trained models saved to `models/`
- Results saved to `results/model_metrics.csv`

### Step 4 — Uncertainty Analysis
**`04_uncertainty_analysis.py`**

- Loads saved Random Forest model
- Computes prediction error statistics (RMSE, mean error, standard deviation)
- Constructs **95% prediction intervals** around model outputs
- Generates figures: prediction vs. actual, error distribution, uncertainty interval

---

## Key Findings

- Random Forest and XGBoost **significantly outperform Linear Regression**, confirming the presence of strong nonlinear fatigue mechanisms that S-N curves cannot model
- `stress_range`, `stress_amplitude`, and `mean_stress` are the **dominant predictors** of fatigue life, consistent with classical fatigue mechanics
- **Log transformation** of fatigue life effectively reduces distributional skewness and improves model accuracy
- **95% prediction intervals** provide a foundation for probabilistic reliability assessment of welded bridge components

---

## Getting Started

All notebooks are designed to run on **Google Colab** with Google Drive mounted at:

```
/content/drive/MyDrive/Fatigue-Life-ML/
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run notebooks in order: `01 → 02 → 03 → 04`

---

## References

- Deng, C. et al. "A Dataset of Fatigue Properties for Welded Joints." *Scientific Data, Nature*, November 2025. https://doi.org/10.6084/m9.figshare.29254265.v2
- IIW Recommendations for Fatigue Design of Welded Joints and Components, 2nd Ed.
- Eurocode 3 — Design of Steel Structures, Part 1-9: Fatigue

---

## Author

**Monisha Patnana**
B.Tech — Civil / Structural Engineering
2026
