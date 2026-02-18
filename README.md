# housing-price-regression
Comparative analysis of multiple machine learning models for housing price prediction


Regression Project – Numerical Prediction

1. Project Description

This project aims to predict a continuous target variable from a dataset. Two main models were tested: Decision Tree Regressor and Support Vector Regressor (SVR). The workflow includes data preprocessing, feature selection, model training, and performance evaluation.

2. Dataset

The dataset contains numerical values with some missing entries. Example sample:

-3.78818017e+09, -3.75020745e+09, -3.78178878e+09, ..., NaN, NaN

Missing values were handled during preprocessing to allow effective model training.

3. Preprocessing

Removal or imputation of missing values.

Normalization / standardization of features for scale-sensitive models (SVM).

Feature engineering, including ratio and interaction features.


4. Models and Hyperparameters

Decision Tree Regressor

max_depth: 5

min_samples_split: 4

min_samples_leaf: 2


Evaluation:

MAE: 49,787.05

MSE: 4,185,017,377.08

R² Score: 0.77


Support Vector Regressor (SVR)

Kernel: RBF

Standardization of features


Evaluation:

MAE: 48,222.13

MSE: 4,079,664,426.04

R² Score: 0.78


5. Visualizations

Feature Importance (Decision Tree): Horizontal bars show relative importance of each feature.

Decision Tree: Limited to 3 levels for clarity, highlighting key splits.

Learning Curves: Compare R² between train and test sets to detect overfitting.

R² train: 0.82

R² test: 0.77



6. Conclusion

Both Decision Tree and SVR show strong performance.

SVR slightly outperforms Decision Tree in accuracy.

The project can be extended with ensemble methods for improved predictions.


7. How to Run the Code

7.1 Install Required Libraries

pip install numpy pandas matplotlib seaborn scikit-learn

7.2 Load the Dataset

import pandas as pd
df = pd.read_csv('your_dataset.csv')

7.3 Run the Notebook or Script

Execute the Python notebook or script to preprocess data, train models, and evaluate results.
