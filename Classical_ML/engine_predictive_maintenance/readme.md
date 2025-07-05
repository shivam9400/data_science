# 🛠️ Predictive Maintenance: RUL Estimation & Failure Prediction

This repository contains an end-to-end machine learning pipeline to predict **Remaining Useful Life (RUL)** and **failure risk** of machines based on time-series sensor and operational data.

The project simulates a real-world **predictive maintenance** scenario in manufacturing or aerospace, where the goal is to anticipate failures **before they occur**.

---

## 📌 Problem Statement

Given time-series data from multiple machines (or engines), predict:

1. The **Remaining Useful Life (RUL)** — number of cycles until failure.
2. A **binary label** indicating whether a failure will occur within the next `N` cycles (e.g., `15` cycles).

---

## 🔍 Dataset Overview

Each row in the dataset represents one cycle from one unit and contains:

- `UnitNumber`: Machine ID
- `Cycle`: Operating cycle count
- `Op_Setting_1/2/3`: Operational conditions
- `Sensor_1` to `Sensor_n`: Sensor readings
- `Target_RUL`: Calculated during preprocessing for regression
- `Target_15_Cycles`: Binary classification label (RUL ≤ 15 → failure risk)

---

## ⚙️ Project Workflow

### 🧹 1. Data Preprocessing

- Compute **RUL** per unit (`max_cycle - cycle`)
- Create **binary label** for classification (`RUL ≤ 15`)
- Detect and remove **leaky features**
- Handle missing values and feature correlations

### 📊 2. Exploratory Data Analysis

- Visualizations using Seaborn: sensor trends vs. RUL
- Feature correlation heatmaps
- Residual plots and model error analysis

### 🔁 3. Regression Model (RUL Prediction)

- Model: `XGBRegressor` with `log1p(RUL)` as target
- **Sample weighting**: More weight on high RUL samples
- Hyperparameter tuning with `GridSearchCV`
- Feature scaling with `StandardScaler`
- Metrics: `MSE`, `MAE`, `R²`

### ⚠️ 4. Classification Model (Failure within 15 cycles)

- Model: `RandomForestClassifier`
- Address class imbalance with `class_weight` and `sample_weight`
- ROC Curve and AUC Score for evaluation
- Metrics: `Accuracy`, `Precision`, `Recall`, `ROC-AUC`

---

## ✅ Results Summary

### 📈 Regression

| Metric        | Score   |
|---------------|---------|
| R² Score      | 0.63    |
| MAE           | 29.6    |
| MSE           | 1755    |
| Model         | XGBoost (log-transformed target, weighted loss) |

### 🧮 Classification

| Metric        | Score   |
|---------------|---------|
| Accuracy      | 97.7%   |
| ROC-AUC       | 99%     |
| Precision     | 81.5%   |
| Recall        | 86.3%   |
