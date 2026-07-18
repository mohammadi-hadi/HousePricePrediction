<div align="center">

# House Price Prediction with Regularized Regression

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Dataset: Kaggle](https://img.shields.io/badge/Dataset-Kaggle-20BEFF.svg)](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

*Linear, Lasso, and Ridge regression on the Kaggle Ames housing data, comparing missing-data strategies and stepwise feature selection.*

</div>

## Overview

This study project models housing sale prices on the Kaggle *House Prices: Advanced Regression Techniques* dataset. The focus is on how preprocessing choices interact with regularized regression: two missing-value strategies (dropping incomplete rows vs. imputation) are crossed with two feature sets (all features vs. a backward-stepwise selection), and each of the four resulting datasets is fitted with linear, Lasso, and Ridge regression.

## Data

Derived from the Kaggle [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) competition (residential home sales in Ames, Iowa). This branch contains the preprocessed model-ready matrices; the raw `train.csv` / `test.csv` and the official `data_description.txt` are on the [`Data`](../../tree/Data) branch.

- `all_data_dropna.csv` / `all_data_dropna_stepwise.csv` — one-hot encoded features after dropping rows with missing values (2,396 rows; full vs. stepwise-selected columns).
- `all_data_impute.csv` / `all_data_impute_stepwise.csv` — the same after imputing missing values (2,919 rows).
- `price_dropna.csv` / `price_impute.csv` — matching log-transformed `SalePrice` targets.

## Methods

Implemented in `final_code02.py` (on the [`code`](../../tree/code) branch):

- **Target handling**: inspection and log transform of `SalePrice` (histogram and normal probability plots before/after).
- **Missing data**: analysis of missingness, followed by the two handling strategies (drop vs. impute).
- **Feature engineering**: backward-stepwise feature selection and one-hot encoding of categorical variables.
- **PCA**: on the scaled features to examine explained variance.
- **Models**: linear, Lasso, and Ridge regression on each of the four dataset variants, evaluated over repeated random 90/10 train/test splits with averaged R², MSE, and MAE.
- **Extension**: a sketch of a simple advisory step that suggests comparable properties based on shared features.

## Results

In the accompanying course report, the best configuration was Lasso regression on the dropped-rows, stepwise-selected dataset:

| Best configuration | R² (train) | R² (test) |
|---|---|---|
| Lasso — dropped rows + stepwise features | ~0.91 | ~0.86 |

Scores are averages over repeated random 90/10 train/test splits.

## Quick Start

```bash
git clone https://github.com/mohammadi-hadi/HousePricePrediction.git
cd HousePricePrediction
pip install -r requirements.txt

git checkout code   # the analysis script lives on this branch
```

1. The `Data` branch additionally holds the raw Kaggle files, if you want to rebuild the preprocessed matrices.
2. Update the hard-coded file paths in `final_code02.py` to point at the CSVs from the `main` branch.
3. Run the script section by section; each model block prints averaged R², MSE, and MAE for its dataset variant.

## Repository Structure

```
main branch
├── all_data_dropna.csv
├── all_data_dropna_stepwise.csv
├── all_data_impute.csv
├── all_data_impute_stepwise.csv
├── price_dropna.csv
├── price_impute.csv
├── requirements.txt
├── LICENSE
└── README.md

code branch:  final_code02.py            # Full analysis script
Data branch:  train.csv, test.csv, data_description.txt
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Contact

- **Hadi Mohammadi** — Utrecht University
- Website: [mohammadi.cv](https://mohammadi.cv)
