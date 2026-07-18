# House Price Prediction

Predicting house sale prices with linear, Lasso, and Ridge regression, comparing missing-data strategies and stepwise feature selection.

[![Python](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

This study project models housing sale prices on the Kaggle *House Prices: Advanced Regression Techniques* dataset. The focus is on how preprocessing choices interact with regularized regression: two missing-value strategies (dropping incomplete rows vs. imputation) are crossed with two feature sets (all features vs. a backward-stepwise selection), and each of the four resulting datasets is fitted with linear, Lasso, and Ridge regression. In the accompanying course report, the Lasso model on the dropped-rows, stepwise-selected data was chosen as the best configuration (reported around 0.91 R² on training and 0.86 on test splits).

## Data

Derived from the Kaggle [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) competition (residential home sales in Ames, Iowa). This branch contains the preprocessed model-ready matrices; the raw `train.csv` / `test.csv` and the official `data_description.txt` are on the [`Data`](../../tree/Data) branch.

- `all_data_dropna.csv` / `all_data_dropna_stepwise.csv` — one-hot encoded features after dropping rows with missing values (2,396 rows; full vs. stepwise-selected columns).
- `all_data_impute.csv` / `all_data_impute_stepwise.csv` — the same after imputing missing values (2,919 rows).
- `price_dropna.csv` / `price_impute.csv` — matching log-transformed `SalePrice` targets.

## Methods

Implemented in `final_code02.py` (on the [`code`](../../tree/code) branch):

- Target inspection and log transform of `SalePrice` (histogram and normal probability plots before/after).
- Missing-data analysis, followed by the two handling strategies (drop vs. impute).
- Backward-stepwise feature selection and one-hot encoding of categorical variables.
- PCA on the scaled features to examine explained variance.
- Linear, Lasso, and Ridge regression on each of the four dataset variants, evaluated over repeated random 90/10 train/test splits with averaged R², MSE, and MAE.
- A sketch of a simple advisory step that suggests comparable properties based on shared features.

## Repository structure

```
main branch
├── all_data_dropna.csv
├── all_data_dropna_stepwise.csv
├── all_data_impute.csv
├── all_data_impute_stepwise.csv
├── price_dropna.csv
├── price_impute.csv
└── README.md

code branch:  final_code02.py            # Full analysis script
Data branch:  train.csv, test.csv, data_description.txt
```

## Requirements / How to run

Python 3 with:

```
pandas numpy scipy matplotlib seaborn scikit-learn
```

1. Check out the `code` branch for `final_code02.py` (and the `Data` branch if you want the raw Kaggle files).
2. Update the hard-coded file paths in the script to point at the CSVs from this repository.
3. Run the script section by section; each model block prints averaged R², MSE, and MAE for its dataset variant.

## License

MIT — see [LICENSE](LICENSE).

## Author

Hadi Mohammadi — [mohammadi.cv](https://mohammadi.cv)
