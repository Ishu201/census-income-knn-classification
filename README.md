# Census Income Classification: KNN Predictive Model

Predicting whether a U.S. citizen earns above or below $50,000 annually
using a K-Nearest Neighbors classifier trained on UCI Adult Census data.

Built as part of ALY 6020: Predictive Analytics at Northeastern University.

---

## Project Overview

Income inequality is a pressing policy challenge, and data-driven tools
can help identify which demographic and occupational attributes correlate
with higher earning potential. This project builds and evaluates a KNN
classification model to distinguish high-income (>50K) from low-income
(<=50K) citizens, with a companion comparison against Decision Tree and
Logistic Regression models.

---

## Dataset

**Source:** [UCI Adult Census Dataset](https://archive.ics.uci.edu/ml/datasets/adult)  
**Records:** 42,468 after preprocessing  
**Features:** 14 attributes including age, education, occupation,
capital-gain, marital status, and more  
**Target:** Binary income classification (<=50K / >50K)  
**Class distribution:** 75.4% low income, 24.6% high income

> The dataset is not included in this repository. Download it from the
> UCI link above and place `adult.data` in a `/data` folder before
> running the notebook.

---

## Methodology

### Part 1: Data Cleansing
- Loaded with `na_values='?'` to handle encoded missing values
- Dropped `fnlwgt` (sampling weight, no predictive value) and redundant
  `education` column (replaced by numeric `education-num`)
- Removed duplicate records
- Outlier capping at 99th percentile for `capital-gain` and
  `capital-loss` using IQR detection
- One-hot encoding for 7 nominal categorical variables (drop_first=True)
- StandardScaler normalization across all 84 features
- Stratified 80/20 train/test split (random_state=42)

### Part 2: KNN Modeling
- Optimal K selected via 5-fold cross-validation across odd values K=1
  to K=30
- Elbow point identified at **K=15**
- Permutation importance used for feature contribution analysis (KNN
  has no native coefficients)

---

## Results

| Model | Accuracy | AUC |
|---|---|---|
| KNN (K=15) | 82.79% | 0.8748 |
| Logistic Regression | 84.24% | 0.9008 |
| Decision Tree | 80.29% | 0.7412 |

**Top predictors (permutation importance):** capital-gain, education-num,
relationship type, age, executive/managerial occupation

---

## Key Findings

- Capital-gain was the strongest predictor by a large margin, pointing
  to existing wealth as the primary structural barrier between income
  groups
- Education level is the most policy-actionable predictor
- Logistic regression achieved the highest AUC (0.9008), suggesting the
  income boundary in this dataset is largely linearly separable
- Class imbalance (75/25) limited minority class recall to 0.55 for
  the >50K group

---

## Tech Stack

- Python 3
- pandas, NumPy
- scikit-learn
- matplotlib, seaborn

---

## Files

| File | Description |
|---|---|
| `census_income_knn.ipynb` | Full analysis notebook |
| `Isuri_Module1_KNN_Report.pdf` | Written report with interpretation |

---

## Future Work

- Apply SMOTE oversampling to address the 75/25 class imbalance
- Explore ensemble methods (Random Forest, Gradient Boosting) to
  improve minority class recall
- Migrate pipeline to Spark MLlib for scalability

---

## Author

**Isuri Nawodya**  
Master of Professional Studies in Analytics, Northeastern University Vancouver  
[LinkedIn](https://linkedin.com/in/isuri-nawodya) |
[GitHub](https://github.com/Ishu201)
