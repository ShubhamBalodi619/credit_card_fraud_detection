# Credit Card Fraud Detection

A machine learning project to detect fraudulent credit card transactions using a variety of classification models, cross-validation strategies, and class imbalance handling techniques.

---

## Dataset

**Source:** [Credit Card Fraud Detection Dataset 2023](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023) by Nidula Elgiriyewithana on Kaggle

- Over **550,000 transaction records** from European cardholders in 2023
- Features **V1–V28** are anonymized to protect cardholder privacy
- `Amount` — transaction amount
- `Time_Hour` — hour of the transaction (derived from `Time`)
- `Class` — binary target label: `1` = Fraudulent, `0` = Legitimate
- Highly imbalanced dataset: only **~0.5% of transactions are fraudulent**

---

## Installation

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost imbalanced-learn
```

---

## Methodology

### 1. Exploratory Data Analysis
- Class distribution analysis (pie chart, bar chart)
- Correlation heatmap
- Feature distribution plots (normal vs fraud overlay)

### 2. Cross-Validation Strategies
| Strategy | Config |
|----------|--------|
| RepeatedKFold | 5 splits, 10 repeats |
| StratifiedKFold | 5 splits, shuffle=True |

### 3. Class Imbalance Handling
| Technique | Description |
|-----------|-------------|
| RandomOverSampler | Duplicates minority class samples (`sampling_strategy=0.5`) |
| SMOTE | Generates synthetic samples between nearest neighbours |
| ADASYN | Adaptively generates more samples in harder-to-classify regions |

### 4. Models
| Model | Notes |
|-------|-------|
| Logistic Regression (L1) | `solver=saga` |
| Logistic Regression (L2) | `solver=newton-cg` |
| K-Nearest Neighbours | `n_neighbors=5` |
| Decision Tree | Gini and Entropy criteria |
| Random Forest | `n_estimators=100`, `max_features=sqrt` |
| XGBoost | Default + hyperparameter tuned |
| SVM | `kernel=sigmoid`, `probability=True` |

### 5. Hyperparameter Tuning
- `RandomizedSearchCV` on XGBoost across 7 hyperparameters: `max_depth`, `min_child_weight`, `n_estimators`, `learning_rate`, `gamma`, `subsample`, `colsample_bytree`

---

## Results

| Rank | Methodology | Model | ROC AUC | Accuracy |
|------|-------------|-------|---------|----------|
| 🥇 1 | RepeatedKFold CV | Logistic Regression L2 | **0.9923** | 0.9989 |
| 🥈 2 | Random Oversampling + StratifiedKFold | Logistic Regression L2 | 0.9823 | 0.9882 |
| 🥉 3 | StratifiedKFold CV | Logistic Regression L2 | 0.9820 | 0.9991 |
| 4 | ADASYN + StratifiedKFold | XGBoost | 0.9815 | 0.9993 |
| 5 | SMOTE + StratifiedKFold | XGBoost | 0.9805 | 0.9994 |

> **Note:** ROC AUC is the primary evaluation metric. Accuracy alone is misleading for highly imbalanced datasets — a model predicting all transactions as legitimate would still achieve ~99.5% accuracy.

### Best Model per Methodology
| Methodology | Best Model | ROC AUC |
|-------------|-----------|---------|
| RepeatedKFold CV | Logistic Regression L2 | **0.9923** |
| StratifiedKFold CV | Logistic Regression L2 | 0.9820 |
| Random Oversampling | Logistic Regression L2 | 0.9823 |
| SMOTE | XGBoost | 0.9805 |
| ADASYN | XGBoost | 0.9815 |

---

## Key Findings

- **Logistic Regression with L2 regularization** using RepeatedKFold cross-validation achieved the best overall ROC AUC of **0.9923**
- **SVM with sigmoid kernel** performed worst (ROC AUC ~0.45–0.48), worse than random guessing — the kernel choice is likely a poor fit for this data
- **L1 Logistic Regression** improved dramatically with oversampling (from 0.567 to 0.964 with ADASYN), suggesting it struggled with class imbalance
- **Oversampling reduced accuracy** for Logistic Regression while improving minority class recall — a worthwhile trade-off for fraud detection
- **XGBoost and Random Forest** were consistently strong across all methodologies

---

## Tech Stack

| Category | Libraries |
|----------|-----------|
| Data Handling | `pandas`, `numpy` |
| Visualisation | `matplotlib`, `seaborn` |
| Machine Learning | `scikit-learn`, `xgboost` |
| Class Imbalance | `imbalanced-learn` |

---
