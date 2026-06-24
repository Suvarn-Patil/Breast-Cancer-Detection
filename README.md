# Breast Cancer Diagnosis Classification: A PRML Case Study

## Abstract

This project presents a full supervised machine learning workflow for binary breast cancer diagnosis using the Wisconsin Diagnostic dataset. The study compares multiple classical classifiers, applies feature selection and dimensionality analysis, performs model tuning with cross-validation, and evaluates clinical relevance through recall, ROC-AUC, and confusion matrices. The implementation is provided as a single reproducible notebook for transparent experimentation.

## 1. Problem Statement

Early and accurate detection of malignant tumors is a high-impact healthcare problem. From a machine learning perspective, this is a binary classification task where the model must distinguish malignant and benign cases from numeric tumor characteristics.

Primary objectives:

- maximize predictive reliability on unseen data
- reduce missed malignant diagnoses (false negatives)
- maintain interpretability suitable for a medical-analysis setting

## 2. Dataset Description

The analysis is performed on the Breast Cancer Wisconsin (Diagnostic) dataset.

- Data file used: `breast+cancer+wisconsin+diagnostic/wdbc.data`
- Target variable: `Diagnosis`
- Label encoding used in notebook:
  - `M -> 1` (Malignant)
  - `B -> 0` (Benign)

Feature groups correspond to computed characteristics of cell nuclei (for example radius, texture, smoothness, compactness, concavity, symmetry, and related derivatives).

## 3. Methodology

### 3.1 Data Preparation

- assign explicit column names to raw dataset
- split features (`X`) and labels (`y`)
- convert diagnosis labels into binary numeric format

### 3.2 Exploratory Data Analysis

The notebook includes structured EDA to understand class behavior and feature interactions:

- class-distribution visualization
- boxplots and histograms across key features
- full correlation analysis (including masked triangular heatmap)
- pairplot and violin plots for selected high-signal features

### 3.3 Preprocessing and Leakage Control

To support robust evaluation, the workflow includes:

- removal of perfectly/redundantly correlated variables
- IQR-based outlier capping
- stratified train/test split
- standardization fit on training data and applied to test data

This ordering is designed to prevent optimistic leakage into evaluation metrics.

### 3.4 Feature Selection and Dimensionality Analysis

- ANOVA F-test via `SelectKBest` with `k = 20`
- ranked F-score analysis and visualization
- PCA projection (2D) for structure visualization
- cumulative explained variance analysis across components

### 3.5 Model Development and Selection

Baseline models evaluated with 5-fold cross-validation on training data:

- Logistic Regression
- K-Nearest Neighbors
- Support Vector Machine
- Decision Tree
- Random Forest
- Gradient Boosting
- AdaBoost
- Gaussian Naive Bayes

Hyperparameter tuning is performed using `GridSearchCV` for:

- AdaBoost
- SVM
- Logistic Regression

## 4. Evaluation Protocol

Final model quality is assessed on the held-out test set using:

- accuracy
- recall for malignant class
- classification report
- ROC curve and AUC
- confusion matrix

The workflow emphasizes recall and discrimination quality because clinical tasks are sensitive to malignant-case misses.

## 5. Explainability Component

To improve interpretability, a Random Forest model is trained on selected top features and feature importance is visualized. This provides a direct view of which tumor characteristics contribute most strongly to malignant predictions.

## 6. Reproducibility

Repository contents:

```text
.
├── prml_project.ipynb
└── README.md
```

### 6.1 Package Requirements

Recommended Python version:

- Python 3.10+

Required libraries (tested baseline versions):

- numpy>=1.24
- pandas>=2.0
- seaborn>=0.12
- matplotlib>=3.7
- scikit-learn>=1.3
- ucimlrepo>=0.0.7
- jupyter>=1.0

Install all dependencies:

```bash
pip install "numpy>=1.24" "pandas>=2.0" "seaborn>=0.12" "matplotlib>=3.7" "scikit-learn>=1.3" "ucimlrepo>=0.0.7" "jupyter>=1.0"
```

### 6.2 Run Instructions

1. Clone or download this repository.
2. Create and activate a Python virtual environment.
3. Install the required packages using the command above.
4. Ensure the dataset file exists at `breast+cancer+wisconsin+diagnostic/wdbc.data`.
5. Start Jupyter:

  ```bash
  jupyter notebook
  ```

6. Open `prml_project.ipynb` in the browser.
7. Run notebook cells from top to bottom to reproduce results.

## 7. Limitations and Scope

- The project is a methodological PRML study and not a clinical deployment system.
- Results depend on a single dataset and split strategy.
- Variable naming in a few notebook cells reflects iterative experimentation and may be standardized further.

## 8. Future Work

- add a consolidated metrics comparison table for tuned models
- package preprocessing and estimator into one sklearn pipeline object
- perform probability calibration analysis
- save and version the final selected model for reproducible inference

## 9. Conclusion

This project demonstrates a complete and careful machine learning pipeline for medical binary classification, combining statistical feature analysis, model benchmarking, hyperparameter tuning, and interpretable evaluation. The notebook is structured to balance predictive performance with transparent, report-ready analysis.
