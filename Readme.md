# Short-Horizon Stock Selection via Machine Learning

Course: DSA5205 - Data Science in Quantitative Finance

Date: November 2025


## 1. Project Overview

This project implements a systematic, short-horizon stock trading strategy driven by machine learning. The core objective is to predict next-day open-to-open returns for a universe of seven liquid U.S. technology stocks and allocate capital using a volatility-adjusted portfolio approach.

The project emphasizes a rigorous backtesting framework that strictly avoids look-ahead bias, incorporates realistic transaction costs, and compares linear vs. non-linear modeling approaches.

**Target**: Next-day open-to-open returns ($P_{t+1, open} / P_{t, open} - 1$).

**Universe**: AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA.

**Data Period**: Jan 1, 2015 – Jan 1, 2019 (Training: 80%, Testing: 20%).

**Methodology**: Feature engineering (Momentum, Volatility, ATR) $\to$ Rolling Normalization $\to$ Predictive Modeling $\to$ Portfolio Optimization.

## 2. Team Members

| Name | Student ID | Role |
| :--- | :--- | :--- |
| **Ziyu Wang** | A0326584W | Data Processing & Feature Engineering |
| **Hongyu Zhu** | A0326180L | Strategy Logic & Portfolio Construction |
| **Deng Boyin** | A0326565X | Backtesting & Performance Analysis |
| **Yujie Cai** | A0327508B | Model Selection, Training & Validation |
| **Kaining Cao** | A0329218B | Model Selection, Training & Validation |

## 3. Installation & Requirements

The project is implemented in Python. Ensure you have the following libraries installed:

```
pip install -r requirements.txt
```

Core Dependencies Explain:

**yfinance**: For downloading historical OHLCV data.

**scikit-learn**: For linear models (Ridge, Lasso), MLP, and rolling validation.

**scipy.optimize**: For Mean-Variance Optimization (SLSQP).

**xgboost**: For non-linear model comparison.

## 4. Methodology

### 4.1 Data Pipeline

Source: Yahoo Finance API (yfinance).

Preprocessing: Auto-adjusted for dividends and splits.

Feature Engineering: Constructed 380+ features including Momentum (5d, 20d), Volatility Dynamics (RV, ATR), and Volume Ratios for each window.

Sanitization: Rolling window winsorization (1st/99th percentile) and Z-score normalization (252-day lookback) to prevent look-ahead bias.

### 4.2 Model Selection

We evaluated both linear and non-linear models using time-series cross-validation:

Linear: Ridge Regression (L2), Lasso Regression (L1).

Non-Linear: MLP (Neural Net), PCA+MLP, Random Forest, XGBoost, LightGBM, SVR.

Selected Model: Ridge Regression ($\alpha=10000$).

Reasoning: In a low signal-to-noise ratio environment, complex non-linear models suffered from severe overfitting (negative $R^2$ OOS). Ridge regression provided the most robust, albeit modest, predictive power.

### 4.3 Portfolio Strategy

We implemented and compared two allocation methods based on Ridge predictions:

Ridge + MVO: Mean-Variance Optimization maximizing the Sharpe Ratio (Rolling 20-day covariance).

Pure Ridge (Constrained): Proportional allocation based on positive signal strength (capped at 50% weight).


## 6. Conclusion
The Pure Ridge strategy demonstrated superior robustness compared to the MVO approach, which was sensitive to estimation errors in the covariance matrix and expected returns.

## 7. Usage
To reproduce the results, simply download all dependencies and run all cells. This will yield the exact same results as presented in the report.

## 8. File Structure
```
├── DSA5205_team3.ipynb     # Main notbook
├── README.md               # Project documentation
└── requirements.txt        # Python dependencies
```