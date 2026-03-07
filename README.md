# Fatigue Life Prediction of Welded Joints in Bridge Structures
### Machine Learning for Structural Reliability Assessment

**A data-driven machine learning framework for predicting the fatigue life of welded bridge joints, developed using 45,315 experimental observations from a peer-reviewed fatigue database (Figshare, 2025).**

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Pipeline](#pipeline)
- [Models](#models)
- [Key Findings](#key-findings)
- [Applications](#applications)
- [Getting Started](#getting-started)
- [References](#references)
- [Author](#author)

---

## Overview

Fatigue failure is one of the primary causes of long-term structural deterioration in welded bridge components. Conventional assessment approaches rely on empirical S-N curves, which fail to account for the **nonlinear interactions** between weld geometry, material heterogeneity, residual stresses, and variable-amplitude loading histories.

This project addresses that limitation by developing a complete machine learning pipeline — from raw experimental fatigue data through to probabilistic life prediction. The framework integrates structural engineering domain knowledge with modern ML techniques to:

- Accurately estimate fatigue life beyond the capabilities of traditional S-N methods
- Identify the most influential structural and loading parameters
- Quantify prediction uncertainty for reliability-aware decision making
- Support data-informed maintenance strategies for welded bridge structures

---

## Dataset

| Property | Details |
|----------|---------|
| **Source** | Deng, C. et al. *Scientific Data, Nature* (2025) |
| **Link** | https://figshare.com/articles/dataset/A_Dataset_of_Fatigue_Properties_for_Welded_Joints/29254265 |
| **DOI** | https://doi.org/10.6084/m9.figshare.29254265.v2 |
| **Size** | 45,315 stress-life entries |
| **Origin** | Compiled from peer-reviewed publications using NLP, image recognition, and table parsing — as reported by dataset authors (Deng, C. et al., 2025) |

### Key Features

| Category | Features |
|----------|---------|
| **Stress Variables** | Stress range, stress amplitude, mean stress, normalized stress |
| **Material Properties** | Base material yield strength, ultimate strength |
| **Geometric Parameters** | Specimen thickness, weld configuration |
| **Loading Conditions** | Load ratio, variable amplitude loading |
| **Process Parameters** | Welding method, residual stress |

**Target Variable:** `log_fatigue_life` — log₁₀ of fatigue life in cycles to failure

**Note:** Log transformation is applied to reduce distributional skewness and improve model stability across the wide range of fatigue life values.

---

## Project Structure

```
Fatigue-Life-ML/
├── data/
│   ├── raw/
│   │   └── Weld-Fatigue-Database-json/
│   │       ├── parameter.json           # Welding process and material parameters
│   │       └── S-N.json                 # Stress-life experimental results
│   └── processed/
│       └── fatigue_dataset_clean.csv    # Merged and engineered dataset
├── notebooks/
│   ├── 01_data_preprocessing.ipynb      # Data loading, merging, feature engineering
│   ├── 02_eda_analysis.py               # Exploratory data analysis and visualisation
│   ├── 03_model_training.py             # Model training, evaluation, comparison
│   └── 04_uncertainty_analysis.py       # Prediction intervals and error analysis
├── models/
│   ├── linear_regression.pkl            # Saved baseline model
│   ├── random_forest.pkl                # Saved Random Forest model
│   └── xgboost_model.pkl                # Saved XGBoost model
├── results/
│   ├── figures/                         # All saved plots (11 figures)
│   └── model_metrics.csv                # RMSE, MAE, R² for all models
├── requirements.txt
└── README.md
```

---

## Pipeline

```
Data Preprocessing  →  Exploratory Data Analysis  →  Model Training  →  Uncertainty Analysis
```

---

### Step 1 — Data Preprocessing
**`01_data_preprocessing.ipynb`**

The raw fatigue database is provided in two JSON files — one containing welding process and material parameters, the other containing stress-life experimental results. These are merged and enriched with domain-informed derived features.

- Loads and parses `parameter.json` and `S-N.json` using `pd.json_normalize()`
- Merges both files on `dataset_id`
- Handles missing values and enforces data types
- Engineers derived features:
  - `stress_amplitude` = stress_range / 2
  - `mean_stress` = (max stress + min stress) / 2
  - `normalized_stress` = stress_range / yield_strength
  - `log_fatigue_life` = log₁₀(fatigue_life)
- Saves clean dataset to `data/processed/fatigue_dataset_clean.csv`

---

### Step 2 — Exploratory Data Analysis
**`02_eda_analysis.py`**

Thorough visual and statistical investigation of the dataset. Each analysis is directly motivated by fatigue mechanics theory.

| Figure | Description |
|--------|-------------|
| **Fatigue Life Distribution** | Confirms log-normal behaviour — justifies log-scale target variable |
| **S-N Curve** | Validates inverse stress-life relationship consistent with Basquin's law |
| **Correlation Heatmap** | Identifies multicollinearity among stress-derived features |
| **Welding Method vs Fatigue Life** | Reveals process-dependent performance variability |
| **Residual Stress vs Fatigue Life** | Examines influence of thermally-induced residual stresses |
| **Outlier Detection** | Retains extreme values — consistent with genuine fatigue scatter |
| **Stress Feature Distributions** | Illustrates wide loading range represented in the dataset |
| **Material Property Influence** | Examines how base material yield and ultimate strength affect fatigue life |
| **Geometry Influence** | Analyses the effect of specimen thickness and weld configuration on fatigue performance |

All figures saved to `results/figures/`.

---

### Step 3 — Model Training and Comparison
**`03_model_training.py`**

Three regression models are trained and benchmarked, spanning from a linear baseline to advanced nonlinear ensemble methods.

| Model | Type | Purpose |
|-------|------|---------|
| **Linear Regression** | Linear | Baseline — establishes minimum performance threshold |
| **Random Forest** | Nonlinear ensemble | Captures complex feature interactions via bagged decision trees |
| **XGBoost** | Gradient boosting | High-performance sequential boosting, robust to noise |

- **Metrics:** RMSE, MAE, R²
- Trained models saved to `models/` as `.pkl` files
- Results saved to `results/model_metrics.csv`
- Feature importance plot identifies top 15 predictors of fatigue life

---

### Step 4 — Uncertainty Analysis
**`04_uncertainty_analysis.py`**

Quantifies the reliability of model predictions — essential for structural engineering applications where safety margins must be understood.

- Loads the saved Random Forest model
- Computes RMSE, mean prediction error, and error standard deviation
- Computes prediction intervals around model outputs using error standard deviation
- Produces three diagnostic figures:
  - Prediction vs. Actual scatter plot with reference diagonal
  - Error distribution histogram (bias check)
  - Uncertainty interval visualisation

---

## Models

| Model | RMSE | MAE | R² |
|-------|------|-----|----|
| Linear Regression | 0.7230 | 0.5492 | 0.3818 |
| Random Forest | 0.4891 | 0.3383 | 0.7171 |
| XGBoost | 0.6359 | 0.4792 | 0.5218 |

**Random Forest achieves the best performance (R² = 0.717, RMSE = 0.489)** — confirming its superior ability to capture nonlinear fatigue behaviour over Linear Regression and XGBoost.

---

## Key Findings

- **Random Forest and XGBoost significantly outperform Linear Regression** — confirming strong nonlinear fatigue mechanisms that conventional S-N curves cannot model
- **`stress_range`, `stress_amplitude`, and `mean_stress`** are the dominant predictors, consistent with classical fatigue mechanics and Basquin's power law
- **Log transformation** of fatigue life effectively reduces skewness and substantially improves model fit
- **95% prediction intervals** quantify model prediction uncertainty, providing insight into the confidence of fatigue life estimates for welded bridge components

---

## Applications

This framework can directly support:

- **Bridge structural health monitoring** — continuous fatigue life tracking under real service loads
- **Fatigue reliability assessment** — probabilistic estimation of remaining structural life
- **Infrastructure maintenance planning** — data-driven prioritisation of inspection and repair
- **AI-assisted structural engineering** — interpretable ML predictions grounded in domain knowledge

---

## Getting Started

All notebooks are designed to run on **Google Colab** with Google Drive mounted at:

```
/content/drive/MyDrive/Fatigue-Life-ML/
```

**1. Clone the repository:**

```bash
git clone https://github.com/Monisha325/fatigue-life-ml.git
cd fatigue-life-ml
```

**2. Install dependencies:**

```bash
pip install -r requirements.txt
```

**3. Run notebooks in order:**

```
01_data_preprocessing.ipynb
02_eda_analysis.py
03_model_training.py
04_uncertainty_analysis.py
```

---

## References

- Deng, C. et al. "A Dataset of Fatigue Properties for Welded Joints." *Scientific Data, Nature*, November 2025. https://doi.org/10.6084/m9.figshare.29254265.v2
- IIW Recommendations for Fatigue Design of Welded Joints and Components, 2nd Edition
- Eurocode 3 — Design of Steel Structures, Part 1-9: Fatigue

---

## Author

**Monisha Patnana**

B.Tech — Computer Science and Engineering (AI & ML)

Machine Learning Enthusiast

2026
