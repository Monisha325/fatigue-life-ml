# 🔩 Fatigue Life Prediction of Welded Bridge Joints

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Platform-Google%20Colab-orange?logo=googlecolab&logoColor=white" />
  <img src="https://img.shields.io/badge/ML-scikit--learn%20%7C%20XGBoost-green?logo=scikitlearn&logoColor=white" />
  <img src="https://img.shields.io/badge/Dataset-45%2C315%20samples-lightgrey" />
  <img src="https://img.shields.io/badge/Best%20R²-0.717%20(Random%20Forest)-success" />
  <img src="https://img.shields.io/badge/License-MIT-yellow" />
</p>

> **Machine Learning for Structural Reliability Assessment** — A data-driven framework for predicting the fatigue life of welded joints in bridge structures, developed using 45,315 experimental observations from a peer-reviewed fatigue database.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Research Motivation](#-research-motivation)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Pipeline](#-pipeline)
  - [Step 1 — Data Preprocessing](#step-1--data-preprocessing)
  - [Step 2 — Exploratory Data Analysis](#step-2--exploratory-data-analysis)
  - [Step 3 — Model Training](#step-3--model-training)
  - [Step 4 — Uncertainty Analysis](#step-4--uncertainty-analysis)
- [Models & Results](#-models--results)
- [Key Findings](#-key-findings)
- [Applications](#-applications)
- [Getting Started](#-getting-started)
- [Limitations & Future Work](#-limitations--future-work)
- [References](#-references)
- [Acknowledgements](#-acknowledgements)
- [Author](#-author)

---

## 🔍 Overview

Fatigue failure is one of the primary causes of long-term structural deterioration in welded bridge components. Conventional assessment methods rely on empirical **S-N curves**, which fail to capture the nonlinear interactions between:

- Weld geometry and configuration
- Material heterogeneity (yield strength, ultimate strength)
- Thermally-induced residual stresses
- Variable-amplitude loading histories

This project addresses those limitations through a **complete machine learning pipeline** — from raw experimental fatigue data through to probabilistic life prediction — integrating structural engineering domain knowledge with modern ML techniques.

---

## 🎯 Research Motivation

> *"This project investigates the application of Artificial Intelligence for predicting the fatigue life of welded joints in bridge structures, addressing limitations inherent in conventional analytical and experimental assessment methods."*

By developing data-driven predictive models, this research aims to:

- **Accurately estimate fatigue life** beyond the capabilities of traditional S-N curve methods
- **Capture nonlinear fatigue mechanisms** that conventional approaches cannot model
- **Quantify prediction uncertainty** for reliability-aware structural decision making
- **Support informed maintenance strategies** for extending the operational lifespan of welded bridge components

---

## 📊 Dataset

| Property | Details |
|----------|---------|
| **Source** | Deng, C. et al. *Scientific Data*, Nature (2025) |
| **Link** | [Figshare Repository](https://figshare.com/articles/dataset/A_Dataset_of_Fatigue_Properties_for_Welded_Joints/29254265) |
| **DOI** | https://doi.org/10.6084/m9.figshare.29254265.v2 |
| **Size** | 45,315 stress-life experimental entries |
| **Format** | Two JSON files: `parameter.json` + `S-N.json` |
| **Origin** | Compiled from peer-reviewed publications using NLP, image recognition, and table parsing |

### Features Used

| Category | Features |
|----------|---------|
| **Stress Variables** | `stress_range`, `stress_amplitude`, `mean_stress`, `normalized_stress`, `stress_strength_ratio` |
| **Material Properties** | `base_yield_strength`, `base_ultimate_strength` |
| **Geometric Parameters** | `fatigue_specimen_thickness`, `welding_joint`, `fatigue_specimen_type` |
| **Loading Conditions** | `load_ratio`, `stress_concentration` |
| **Process Parameters** | `welding_method`, `welding_voltage`, `welding_current`, `welding_speed`, `residual_stress` |
| **Target Variable** | `log_fatigue_life` — log₁₀ of fatigue life (cycles to failure) |

> **Note:** Log₁₀ transformation is applied to the fatigue life target to reduce distributional skewness and improve model stability across the wide dynamic range of fatigue life values.

---

## 🗂️ Project Structure

```
Fatigue-Life-ML/
├── data/
│   ├── raw/
│   │   └── Weld-Fatigue-Database-json/
│   │       ├── parameter.json           # Welding process & material parameters
│   │       └── S-N.json                 # Stress-life experimental results
│   └── processed/
│       └── fatigue_dataset_clean.csv    # Merged, cleaned & engineered dataset
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb      # Data loading, merging, feature engineering
│   ├── 02_eda_analysis.ipynb            # Exploratory data analysis & visualisation
│   ├── 03_model_training.ipynb          # Model training, evaluation, comparison
│   └── 04_uncertainty_analysis.ipynb    # Prediction intervals & error diagnostics
│
├── models/
│   ├── linear_regression.pkl            # Saved baseline model
│   ├── random_forest.pkl                # Saved Random Forest model
│   └── xgboost_model.pkl                # Saved XGBoost model
│
├── results/
│   ├── figures/                         # All saved plots (11 figures)
│   └── model_metrics.csv                # RMSE, MAE, R² for all models
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Pipeline

```
Data Preprocessing  ──►  Exploratory Data Analysis  ──►  Model Training  ──►  Uncertainty Analysis
```

### Step 1 — Data Preprocessing
**`01_data_preprocessing.ipynb`**

The raw fatigue database is provided in two JSON files — one containing welding process and material parameters, the other containing stress-life experimental results. Both files are merged and enriched with domain-informed derived features.

**Operations performed:**
- Loads and parses `parameter.json` and `S-N.json` using `pd.json_normalize()`
- Merges both files on `dataset_id`
- Extracts numeric values from mixed-format string fields using regex
- Handles missing values: **median imputation** for numerical columns; `"Unknown"` fill for categoricals
- Removes duplicate records
- Applies one-hot encoding for categorical variables during model preparation

**Feature Engineering:**

| Derived Feature | Formula | Motivation |
|----------------|---------|-----------|
| `stress_amplitude` | `stress_range / 2` | Classical fatigue parameter |
| `mean_stress` | `stress_range × (1 + load_ratio) / 2` | Accounts for stress ratio effect |
| `normalized_stress` | `stress_range / base_yield_strength` | Non-dimensionalises loading against material capacity |
| `stress_strength_ratio` | `stress_range / base_ultimate_strength` | Relates cyclic loading to material failure threshold |
| `log_fatigue_life` | `log₁₀(fatigue_life)` | Target transformation — reduces skewness |

**Output:** `data/processed/fatigue_dataset_clean.csv`

---

### Step 2 — Exploratory Data Analysis
**`02_eda_analysis.ipynb`**

Thorough visual and statistical investigation of the cleaned dataset, with each analysis grounded in fatigue mechanics theory.

| Figure | Description | Insight |
|--------|------------|---------|
| Fatigue Life Distribution | Histogram of `log_fatigue_life` | Confirms log-normal behaviour — justifies log-scale target |
| S-N Curve | Stress range vs. log fatigue life scatter | Validates inverse stress-life relationship consistent with Basquin's law |
| Correlation Heatmap | Feature correlation matrix | Identifies multicollinearity among stress-derived features |
| Welding Method vs Fatigue Life | Boxplot by welding group | Reveals process-dependent fatigue performance variability |
| Residual Stress vs Fatigue Life | Scatter plot | Examines influence of thermally-induced residual stresses |
| Outlier Detection | Boxplot on `log_fatigue_life` | Extreme values retained — consistent with genuine fatigue scatter |
| Stress Feature Distributions | Histograms of 4 stress features | Illustrates wide loading range; right-skewed distributions typical of fatigue data |
| Material Strength vs Fatigue Life | Yield strength scatter | No strong single-variable trend — confirms need for multivariate ML approach |
| Specimen Thickness vs Fatigue Life | Thickness scatter | Thickness alone insufficient; most observations at lower thicknesses |

> Welding methods are grouped into: **Arc Welding** (MIG/TIG/Arc), **Laser Welding**, **Hybrid Welding**, and **Other** — using domain-informed string classification.

All figures saved to `results/figures/`.

---

### Step 3 — Model Training
**`03_model_training.ipynb`**

Three regression models are trained and benchmarked, spanning a linear baseline to advanced nonlinear ensemble methods.

**Train/Test Split:** 80% training / 20% testing (`random_state=42`)

| Model | Configuration | Purpose |
|-------|--------------|---------|
| **Linear Regression** | Default | Baseline — establishes minimum performance threshold |
| **Random Forest** | `n_estimators=200`, `random_state=42`, `n_jobs=-1` | Captures complex feature interactions via bagged decision trees |
| **XGBoost** | `n_estimators=500`, `lr=0.05`, `max_depth=6`, `subsample=0.8` | High-performance sequential gradient boosting, robust to noise |

**Evaluation Metrics:** RMSE, MAE, R²

Trained models saved as `.pkl` files using `joblib`. Feature importance extracted from the Random Forest model (top 15 predictors).

---

### Step 4 — Uncertainty Analysis
**`04_uncertainty_analysis.ipynb`**

Quantifies the reliability of model predictions — critical for structural engineering applications where safety margins must be understood.

**Approach:**
- Loads the saved Random Forest model
- Computes RMSE, mean prediction error, and error standard deviation on the held-out test set
- Constructs **95% prediction intervals** using: `ŷ ± 1.96 × σ_error`

**Diagnostic Figures:**

| Figure | Purpose |
|--------|---------|
| Prediction vs. Actual scatter | Alignment with the ideal diagonal — checks systematic bias |
| Error distribution histogram | Validates near-zero mean error — confirms no strong systematic over/under-prediction |
| Prediction uncertainty interval | Visualises confidence bounds around model outputs |

---

## 📈 Models & Results

| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| Linear Regression | 0.7230 | 0.5492 | 0.3818 |
| **Random Forest** | **0.4891** | **0.3383** | **0.7171** |
| XGBoost | 0.6359 | 0.4792 | 0.5218 |

> **Random Forest achieves the best performance** (R² = 0.717, RMSE = 0.489), confirming its superior ability to capture nonlinear fatigue behaviour compared to both Linear Regression and XGBoost under the current configuration.

---

## 💡 Key Findings

1. **Nonlinear models significantly outperform the linear baseline** — Random Forest and XGBoost both surpass Linear Regression, confirming that fatigue mechanisms in welded joints cannot be adequately modelled by linear S-N approximations.

2. **Dominant predictors are stress-related variables** — `stress_range`, `stress_amplitude`, and `mean_stress` rank highest in feature importance, consistent with Basquin's power law and classical fatigue mechanics theory.

3. **Log transformation is essential** — Applying log₁₀ to the target variable substantially reduces skewness and improves model fit across the wide dynamic range of fatigue life values.

4. **Material yield strength alone is insufficient** — Scatter plots confirm that no single variable strongly determines fatigue life; multivariate interactions captured by ensemble models are necessary.

5. **Prediction uncertainty is quantifiable** — 95% prediction intervals constructed from error standard deviation provide confidence bounds on fatigue life estimates, supporting probabilistic structural reliability assessment.

6. **Welding process matters** — Boxplot analysis shows variability in fatigue performance across welding methods, reflecting differences in heat input, residual stress, and weld quality.

---

## 🏗️ Applications

This framework can directly support:

- **Bridge Structural Health Monitoring** — Continuous fatigue life tracking under real-world service loads
- **Fatigue Reliability Assessment** — Probabilistic estimation of remaining structural life for maintenance scheduling
- **Infrastructure Maintenance Planning** — Data-driven prioritisation of inspection and repair interventions
- **AI-Assisted Structural Engineering** — Interpretable ML predictions grounded in domain-established fatigue mechanics

---

## 🚀 Getting Started

All notebooks are designed to run on **Google Colab** with Google Drive mounted at:
```
/content/drive/MyDrive/Fatigue-Life-ML/
```

### 1. Clone the Repository

```bash
git clone https://github.com/Monisha325/fatigue-life-ml.git
cd fatigue-life-ml
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Core dependencies:**
```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
joblib
```

### 3. Download the Dataset

Download the raw JSON files from [Figshare](https://figshare.com/articles/dataset/A_Dataset_of_Fatigue_Properties_for_Welded_Joints/29254265) and place them at:
```
data/raw/Weld-Fatigue-Database-json/
├── parameter.json
└── S-N.json
```

### 4. Run Notebooks in Order

```
01_data_preprocessing.ipynb   →   Generates fatigue_dataset_clean.csv
02_eda_analysis.ipynb          →   Generates all EDA figures
03_model_training.ipynb        →   Trains and saves all models + model_metrics.csv
04_uncertainty_analysis.ipynb  →   Generates uncertainty diagnostic figures
```

---

## Future Work

- Apply **Bayesian optimisation** or **cross-validated hyperparameter search** for XGBoost and Random Forest
- Investigate **Conformal Prediction** or **Quantile Regression Forests** for statistically valid prediction intervals
- Incorporate **deep learning models** (e.g., feedforward neural networks, gradient-boosted neural networks) for comparison
- Extend to **variable-amplitude loading** fatigue life prediction using damage accumulation rules
- Develop a **web application interface** for engineers to input joint parameters and receive fatigue life estimates with uncertainty bounds

---

## 📚 References

1. Deng, C. et al. *"A Dataset of Fatigue Properties for Welded Joints."* Scientific Data, Nature, 2025. https://doi.org/10.6084/m9.figshare.29254265.v2
2. IIW Recommendations for Fatigue Design of Welded Joints and Components, 2nd Edition. International Institute of Welding.
3. Eurocode 3 — Design of Steel Structures, Part 1-9: Fatigue. European Committee for Standardisation.
4. Basquin, O.H. *"The exponential law of endurance tests."* Proceedings of ASTM, Vol. 10, 1910.

---

## 🙏 Acknowledgements

- Dataset compiled and published by **Deng, C. et al.** via *Scientific Data*, Nature (2025), using NLP, image recognition, and table parsing techniques applied to peer-reviewed fatigue literature.
- Project developed on **Google Colab** using open-source scientific Python libraries.

---

## 👩‍💻 Author

**Monisha Patnana**

B.Tech — Computer Science and Engineering (Artificial Intelligence & Machine Learning)

Machine Learning Enthusiast | Structural AI Applications

---

<p align="center">
  <i>Built with domain knowledge in structural engineering and data-driven machine learning — 2026</i>
</p>
