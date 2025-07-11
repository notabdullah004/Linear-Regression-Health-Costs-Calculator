# Linear-Regression-Health-Costs-Calculator
Linear Regression Health Costs Calculator  Machine Learning with Python
# Linear Regression Health Costs Calculator

This project implements a simple machine learning model to predict individual healthcare costs using linear regression. It is based on the [FreeCodeCamp Linear Regression Health Costs Calculator challenge](https://www.freecodecamp.org/learn/machine-learning-with-python/machine-learning-with-python-projects/linear-regression-health-costs-calculator).

---

## Project Overview

Healthcare costs can vary significantly based on personal factors like age, sex, BMI, smoking status, and region. This project uses a dataset of insurance costs to build a predictive model that estimates expenses based on these features.

---

## Dataset

The dataset includes the following columns:

- `age`: Age of the primary beneficiary
- `sex`: Gender (`male` or `female`)
- `bmi`: Body mass index
- `children`: Number of children/dependents covered by insurance
- `smoker`: Smoking status (`yes` or `no`)
- `region`: Residential area (`northeast`, `northwest`, `southeast`, `southwest`)
- `expenses`: Yearly medical expenses billed by health insurance

The data is publicly available and sourced from [FreeCodeCamp](https://cdn.freecodecamp.org/project-data/health-costs/insurance.csv).

---

## Features

- Data preprocessing using one-hot encoding for categorical features
- Feature scaling with StandardScaler for improved regression performance
- Train-test split (80/20) to validate the model
- Linear Regression model from scikit-learn
- Evaluation with Mean Absolute Error (MAE)
- Visualization of true vs predicted expenses

---

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- matplotlib

You can install the required packages with:

```bash
pip install pandas numpy scikit-learn matplotlib
