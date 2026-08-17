# Rainfall Prediction using Machine Learning & PyCaret AutoML

An end-to-end Machine Learning pipeline and research project designed to predict whether it will rain tomorrow in Australia based on approximately 10 years of daily meteorological observations from various weather stations across the country.

---

## 📌 Table of Contents
1. [Project Overview](#-project-overview)
2. [What is the `.pkl` File?](#-what-is-the-pkl-file)
3. [Dataset Description](#-dataset-description)
4. [Step-by-Step / Cell-by-Cell Notebook Breakdown](#-step-by-step--cell-by-cell-notebook-breakdown)
5. [Model Performance Summary](#-model-performance-summary)
6. [Installation & Setup Guide](#-installation--setup-guide)
7. [How to Load and Use the Saved Model](#-how-to-load-and-use-the-saved-model)

---

## 🌧 Project Overview

Weather forecasting is critical for disaster management, flood prevention, transportation logistics, aviation, and agriculture. This project implements a complete data science workflow using the **Australia Weather Dataset** (145,460 observations, 23 attributes) to forecast the binary target variable **`RainTomorrow`** (`Yes`/`1` or `No`/`0`).

### Key Workflow Highlights:
* **Statistical Imputation:** Uses Random Sample Imputation for variables with high missing percentages and Mean Imputation for continuous features.
* **Feature Engineering:** Directional wind mappings, target-frequency location encoding, and datetime component extraction (`Date_month`, `Date_day`).
* **Class Balancing (SMOTE):** Synthetic Minority Over-sampling Technique to balance the class distribution.
* **Model Training & Comparison:** Six supervised learning algorithms evaluated using Confusion Matrices, Classification Reports, and ROC-AUC curves.
* **Hyperparameter Tuning:** 5-fold Stratified Cross-Validation via `GridSearchCV`.
* **AutoML Integration:** Automated benchmarking and LightGBM training using the **PyCaret** low-code library.
* **Model Serialization:** Deployment-ready exported pickle file (`rain_XGBnew_model.pkl`).

---

## 📦 What is the `.pkl` File (`rain_XGBnew_model.pkl`)?

### 1. What does `.pkl` mean?
`.pkl` stands for a **Python Pickle** file. In Python, **Pickle** is the standard library used to serialize and deserialize object hierarchies:
* **Serialization (Pickling):** Converts an in-memory Python object (such as a trained XGBoost classifier) into a byte stream that can be saved onto a hard drive.
* **Deserialization (Unpickling):** Reads the byte stream back from disk and reconstructs the exact Python object with all its learned weights, tree structures, and hyperparameter states.

### 2. Why is `rain_XGBnew_model.pkl` created?
* **Avoid Retraining:** Training machine learning models on large datasets (145,000+ rows) requires significant computation and time. With the `.pkl` file, the trained model is saved once and can be loaded in milliseconds.
* **Deployment & Production Ready:** The `.pkl` file can be embedded into web applications (Streamlit, Flask, FastAPI), mobile backends, or cloud services to provide instant real-time predictions for user inputs without requiring the training dataset.

---

## 📊 Dataset Description

The dataset [`australia weather.csv`](file:///Users/prabhat/Documents/Rainfall-prediction/australia%20weather.csv) contains daily meteorological measurements:

| Feature | Type | Description |
| :--- | :--- | :--- |
| `Date` | Datetime | Date of observation |
| `Location` | Categorical | Weather station location (49 locations across Australia) |
| `MinTemp` / `MaxTemp` | Float (°C) | Minimum and maximum temperatures recorded in 24 hours |
| `Rainfall` | Float (mm) | Precipitation amount recorded for the day |
| `Evaporation` | Float (mm) | Class A pan evaporation in 24 hours |
| `Sunshine` | Float (hours) | Number of bright sunshine hours in the day |
| `WindGustDir` / `WindGustSpeed` | String / Float | Direction and speed (km/h) of the strongest wind gust |
| `WindDir9am` / `WindDir3pm` | String | Wind direction at 9:00 AM and 3:00 PM |
| `WindSpeed9am` / `WindSpeed3pm` | Float (km/h) | Wind speed at 9:00 AM and 3:00 PM |
| `Humidity9am` / `Humidity3pm` | Float (%) | Relative humidity percentage at 9:00 AM and 3:00 PM |
| `Pressure9am` / `Pressure3pm` | Float (hPa) | Atmospheric pressure reduced to mean sea level |
| `Cloud9am` / `Cloud3pm` | Float (oktas) | Cloud fraction covering the sky (0 to 8 oktas) |
| `Temp9am` / `Temp3pm` | Float (°C) | Temperature at 9:00 AM and 3:00 PM |
| `RainToday` | Binary (0/1) | Whether precipitation exceeded 1mm today |
| **`RainTomorrow`** | **Binary (0/1)** | **Target Variable: Will it rain tomorrow?** |

---

## 🔬 Step-by-Step / Cell-by-Cell Notebook Breakdown

The Jupyter Notebook [`Rain_Prediction using PyCaret (2).ipynb`](file:///Users/prabhat/Documents/Rainfall-prediction/Rain_Prediction%20using%20PyCaret%20(2).ipynb) is structured into 11 distinct phases:

### Phase 1: Environment Setup & Data Ingestion (Cells 0–10)
* **Cells 0–5:** Project title, banner images, objectives, and workflow roadmap.
* **Cell 6:** Imports core libraries (`pandas`, `numpy`, `seaborn`, `matplotlib`, `sklearn`, `warnings`).
* **Cells 8–10:** Reads [`australia weather.csv`](file:///Users/prabhat/Documents/Rainfall-prediction/australia%20weather.csv) and displays the initial 145,460 rows and 23 columns.

### Phase 2: Feature Inspection & Variable Categorization (Cells 11–20)
* **Cells 11–15:** Analyzes unique values per column and segments features into:
  * `num_var`: Numeric continuous and discrete variables.
  * `discrete_var`: Variables with $\le 25$ unique values (`Cloud9am`, `Cloud3pm`).
  * `cont_var`: Continuous measurements (`MinTemp`, `MaxTemp`, `Rainfall`, `Pressure`, etc.).
  * `categ_var`: Categorical features (`Location`, `WindGustDir`, `WindDir9am`, `WindDir3pm`, `RainToday`, `RainTomorrow`).
* **Cells 16–20:** Checks missing value counts and percentage proportions across the dataset.

### Phase 3: Missing Data Imputation Strategy (Cells 21–30)
* **Cells 21–24 (Random Sample Imputation):** Features with a high percentage of missing values (`Cloud9am`, `Cloud3pm`, `Evaporation`, `Sunshine`) are imputed by sampling randomly from non-null values to preserve original variance and distributions.
* **Cells 26–30 (Mean Imputation):** Features with lower missing value percentages (`Pressure9am`, `Pressure3pm`, `MinTemp`, `MaxTemp`, `Rainfall`, `WindSpeed`, `Humidity`, `Temp`) are filled with their respective column mean.

### Phase 4: Exploratory Data Analysis & Visualizations (Cells 31–37)
* **Cells 32–33:** Computes Spearman rank correlation matrix across numeric features and plots an annotated correlation heatmap.
* **Cell 34:** Plots frequency histograms for all numeric variables.
* **Cells 35–37:** Generates KDE distribution plots and boxplots for continuous variables to inspect skewness and outlier presence.

### Phase 5: Categorical Encoding & Feature Engineering (Cells 38–56)
* **Cells 38–39:** Maps binary flags `RainToday` and `RainTomorrow` from `{'No', 'Yes'}` to `{0, 1}` and handles missing entries.
* **Cells 40–44:** Encodes 16 compass wind directions (`NNW`, `NW`, `WNW`, ..., `E`) into integers based on mean rainfall probability and imputes remaining nulls with the mode.
* **Cells 47–53:** Performs frequency-ranked target encoding on `Location` (mapping 49 Australian locations to integers `1` to `49`).
* **Cells 54–56:** Parses `Date` to extract `Date_month` and `Date_day`, and drops the original `Date` string column.

### Phase 6: Target Distribution & SMOTE Oversampling (Cells 57–64)
* **Cells 57–58:** Inspects target class balance (~78% No vs. ~22% Yes).
* **Cells 59–64:** Demonstrates **SMOTE** (Synthetic Minority Over-sampling Technique) to generate synthetic samples of minority class (`RainTomorrow = 1`), showing balanced 50%-50% distribution.

### Phase 7: Q-Q Probability Plots (Cells 65–68)
* **Cells 65–68:** Generates Quantile-Quantile (Q-Q) probability plots for continuous features to analyze normality versus standard Gaussian distribution.

### Phase 8: Feature Scaling & Train-Test Split (Cells 69–79)
* **Cells 69–78:** Fits `StandardScaler` on feature matrix `x` to standardize variables to mean 0 and variance 1:
  $$X_{\text{scaled}} = \frac{x - \mu}{\sigma}$$
* **Cell 79:** Splits the standardized dataset into training (80%, 113,754 samples) and testing (20%, 28,439 samples) partitions using `random_state=0`.

### Phase 9: Supervised Machine Learning Models (Cells 80–125)
Six supervised classifiers are trained, evaluated, and compared:

1. **Random Forest Classifier (Cells 81–87):**
   * Trains an ensemble of 100 decision trees.
   * Evaluates Confusion Matrix, Accuracy (**`85.65%`**), Precision, Recall, and ROC Curve (**AUC: 0.88**).
2. **Gaussian Naive Bayes (Cells 88–94):**
   * Probabilistic classification assuming feature independence.
   * Evaluates Accuracy (**`80.89%`**) and ROC Curve (**AUC: 0.83**).
3. **K-Nearest Neighbors (Cells 95–101):**
   * Instance-based learning using $k=3$ nearest neighbors.
   * Evaluates Accuracy (**`82.34%`**) and ROC Curve (**AUC: 0.77**).
4. **XGBoost Classifier (Cells 102–108):**
   * Extreme Gradient Boosting algorithm.
   * Achieves the highest overall test accuracy (**`86.32%`**) and ROC Curve (**AUC: 0.89**).
5. **Logistic Regression & Grid Search (Cells 109–118):**
   * Base Logistic Regression model (**`84.36%`**).
   * 5-fold Stratified Cross-Validation (`GridSearchCV`) over regularization parameter $C \in [10^0, 10^4]$.
   * Tuned Logistic Regression (**`84.36%`**).
6. **Linear Discriminant Analysis (LDA) & Grid Search (Cells 119–125):**
   * Base LDA model (**`84.26%`**).
   * 5-fold Cross-Validation tuning over solvers (`svd`, `lsqr`, `eigen`) and shrinkage parameters.
   * Tuned LDA (**`84.26%`**).

### Phase 10: Model Serialization & Persistence (Cells 126–130)
* **Cells 127–129:** Serializes the champion **XGBoost** model into [`rain_XGBnew_model.pkl`](file:///Users/prabhat/Documents/Rainfall-prediction/rain_XGBnew_model.pkl).
* **Cell 130:** Re-loads the `.pkl` file from disk and validates inference accuracy on the test set.

### Phase 11: AutoML with PyCaret & LightGBM (Cells 131–142)
* **Cells 131–135:** Overview and architectural introduction to the PyCaret AutoML library.
* **Cells 136–137:** Imports `pycaret.classification`.
* **Cell 138 (`setup`):** Initializes the PyCaret experiment pipeline.
* **Cell 139 (`compare_models`):** Automatically benchmarks top algorithms (`LightGBM`, `Random Forest`, `Logistic Regression`, `LDA`, `Naive Bayes`, `Decision Tree`, `KNN`) across 10 cross-validation folds.
* **Cell 140 (`create_model`):** Trains and evaluates a cross-validated `LightGBM` classifier (**Mean CV Accuracy: 84.60%**).
* **Cells 141–142 (`predict_model`):** Generates holdout test predictions with class probabilities and prediction confidence scores.

---

## 🏆 Model Performance Summary

| Model | Test Accuracy | ROC-AUC | Precision (Rain) | Recall (Rain) | F1-Score (Rain) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **XGBoost Classifier (Best)** | **86.32%** | **0.89** | **0.76** | **0.57** | **0.65** |
| **Random Forest** | 85.65% | 0.88 | 0.78 | 0.50 | 0.61 |
| **Logistic Regression (Tuned)** | 84.36% | 0.86 | 0.73 | 0.47 | 0.57 |
| **Linear Discriminant Analysis** | 84.26% | 0.86 | 0.71 | 0.49 | 0.58 |
| **LightGBM (PyCaret AutoML)** | 84.60% | 0.86 | 0.73 | 0.51 | 0.60 |
| **K-Nearest Neighbors ($k=3$)** | 82.34% | 0.77 | 0.63 | 0.51 | 0.56 |
| **Gaussian Naive Bayes** | 80.89% | 0.83 | 0.56 | 0.63 | 0.59 |

---

## 🚀 Installation & Setup Guide

### 1. Prerequisites
* Python 3.9+ installed.
* macOS users: Ensure OpenMP is installed (`brew install libomp`).

### 2. Clone the Repository
```bash
git clone https://github.com/Xclipxz07/Rainfall-prediction.git
cd Rainfall-prediction
```

### 3. Install Dependencies
```bash
pip install pandas numpy scipy scikit-learn xgboost lightgbm imbalanced-learn seaborn matplotlib pycaret jupyter
```

### 4. Launch the Notebook
```bash
jupyter notebook "Rain_Prediction using PyCaret (2).ipynb"
```

---

## 💡 How to Load and Use the Saved Model

You can load and predict with [`rain_XGBnew_model.pkl`](file:///Users/prabhat/Documents/Rainfall-prediction/rain_XGBnew_model.pkl) in any Python script or web app:

```python
import pickle
import numpy as np
import pandas as pd

# 1. Load the serialized model
with open('rain_XGBnew_model.pkl', 'rb') as f:
    model = pickle.load(f)

# 2. Example 23-feature weather input (standardized)
# ['Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 
#  'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm', 'WindSpeed9am', 
#  'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 
#  'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 'RainToday', 'Date_month', 'Date_day']

sample_features = np.array([[
    10, 13.4, 22.9, 0.6, 4.8, 8.5,
    3, 44.0, 1, 4, 20.0,
    24.0, 71.0, 22.0, 1007.7, 1007.1,
    8.0, 7.0, 16.9, 21.8, 0, 12, 1
]])

# 3. Make Prediction
prediction = model.predict(sample_features)[0]
probability = model.predict_proba(sample_features)[0][1]

print(f"Prediction: {'Rain Tomorrow (Yes)' if prediction == 1 else 'No Rain Tomorrow (No)'}")
print(f"Rain Probability: {probability * 100:.2f}%")
```