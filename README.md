# ft_linear_regression

> Linear regression from scratch — gradient descent with feature normalization and animated training visualization.

## Overview

A linear regression model implemented using only NumPy, trained via gradient descent to predict car prices from mileage. Includes min-max feature normalization, multiple evaluation metrics (MSE, RMSE, MAE, R²), and an animated matplotlib visualization showing the regression line converging in real time.

This project was built as part of the 42 school AI curriculum.

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| Language | Python 3.x |
| Core | NumPy |
| Visualization | Matplotlib (animated training) |
| Data | Pandas (CSV loading) |

## Key Features

- Gradient descent with configurable learning rate and iteration count
- Min-max feature normalization with theta denormalization for interpretable predictions
- Four evaluation metrics implemented from scratch: MSE, RMSE, MAE, R² score (validated against scikit-learn)
- Animated matplotlib visualization of the regression line fitting to data over 50k iterations
- Model persistence via pickle (train once, predict anytime)

## Results

| Metric | Value |
|--------|-------|
| Dataset | Car price vs. mileage (km) |
| Task | Univariate regression |
| Normalization | Min-max scaling |
| Training | 50,000 iterations, α = 0.001 |

## Architecture

```
ft_linear_regression/
├── my_linear_regression.py  # MyLinearRegression class (gradient, fit, predict, normalize)
├── train.py                 # Training script — fits model and saves to disk
├── predict.py               # Load model and predict price from mileage input
├── plot.py                  # Animated training visualization
├── test.py                  # Training convergence test
├── other_losses.py          # MSE, RMSE, MAE, R² (from scratch + sklearn comparison)
└── data.csv                 # Car mileage/price dataset
```

## Getting Started

### Prerequisites

```bash
Python 3.x
NumPy
Matplotlib
Pandas
scikit-learn  # for metric validation only
```

### Installation

```bash
git clone https://github.com/sungyongcho/ft_linear_regression.git
cd ft_linear_regression
pip install numpy matplotlib pandas scikit-learn
```

### Usage

```bash
# Train the model
python train.py

# Predict a car price from mileage
python predict.py

# Watch the training animation
python plot.py
```

## What This Demonstrates

- **Gradient Descent**: Implemented iterative optimization with learning rate control and convergence monitoring — the core algorithm behind all neural network training.
- **Feature Engineering**: Built min-max normalization and theta denormalization to handle features at different scales, a prerequisite for stable gradient descent.
- **Evaluation Metrics**: Implemented MSE, RMSE, MAE, and R² from first principles and validated against scikit-learn, demonstrating understanding of model evaluation beyond accuracy.

## License

This project was built as part of the 42 school curriculum.

---

*Part of [sungyongcho](https://github.com/sungyongcho)'s project portfolio.*
