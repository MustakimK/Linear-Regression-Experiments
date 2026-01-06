# Linear-Regression-Experiments

Implementations of linear regression using both **gradient descent** and the **closed-form ordinary least squares (OLS)** solution, with experimental analysis and visualization.

This repository focuses on understanding the mechanics of regression models without relying on high-level ML libraries.

## Overview

This project explores how linear regression behaves under different optimization approaches by implementing all components manually using NumPy.

Two regression tasks are included:
1. Predicting national happiness scores from GDP per capita
2. Predicting abalone age from physical measurements

The goal is to compare analytical solutions with iterative optimization and understand how learning rate, normalization, and convergence affect performance.

## Implementations

### 1. Gradient Descent vs OLS (Happiness Prediction)

- Single-feature linear regression
- Manual feature normalization
- Gradient descent implemented from scratch
- Experiments across multiple learning rates and epoch counts
- Comparison with analytical OLS solution
- Visualization of multiple regression lines and best-fit comparisons

Key focus:
- Convergence behavior
- Sensitivity to learning rate
- Effect of training duration

### 2. Multivariate Linear Regression (Abalone Age Prediction)

- Multi-feature regression with 7 physical attributes
- Manual normalization of train and test splits
- Closed-form OLS training
- Visualization of feature–target relationships
- Evaluation using mean squared error (MSE)

This experiment highlights how regression scales to higher-dimensional feature spaces.

## Key Concepts Demonstrated

- Linear regression from first principles
- Gradient descent optimization
- Feature normalization
- Analytical vs iterative solutions
- Bias–variance considerations
- Experimental evaluation using MSE
- Data visualization for model interpretation
