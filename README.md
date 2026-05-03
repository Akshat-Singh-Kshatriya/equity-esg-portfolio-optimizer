# AI-Enhanced Equity Portfolio Construction & ESG Analytics

## Overview
This repository contains a quantitative data pipeline and portfolio optimization model built in Python. The project focuses on constructing a passive equity portfolio that maximizes risk-adjusted returns while adhering to strict Environmental, Social, and Governance (ESG) criteria and climate risk parameters.

The model extracts live historical market data, filters equities based on simulated ESG metrics, and utilizes Mean-Variance Optimization (MVO) to calculate optimal asset allocation, significantly improving the portfolio's Sharpe Ratio.

## Key Features
* **Automated Data Pipeline:** Extracts historical adjusted close prices using the `yfinance` API.
* **ESG & Climate Risk Filtering:** Pure Pandas-based data manipulation to filter securities meeting high ESG and low climate risk thresholds.
* **Mean-Variance Optimization:** Utilizes `SciPy` to run Sequential Least Squares Programming (SLSQP), replacing equal-weighting with mathematically optimized allocations.
* **Boundary Constraints:** Implements position sizing constraints (e.g., 5% minimum, 40% maximum per asset) to prevent "corner solutions" and ensure true sector diversification.
* **Statistical Analytics:** Calculates Annualized Returns, Volatility, Covariance Matrices, and Sharpe Ratios.

## Tech Stack
* **Language:** Python (Pandas, Numpy)
* **Optimization:** SciPy (`scipy.optimize`)
* **Data Visualization:** Matplotlib, Seaborn
* **APIs:** Yahoo Finance (`yfinance`)

## Results & Impact
During backtesting (2018-2024 timeframe), transitioning the model from a naive equal-weight allocation to the constrained Mean-Variance Optimization model yielded the following improvements:
* Reduced annualized volatility from **34.90%** to **25.31%**.
* Increased annualized return to **26.85%**.
* Improved the Sharpe Ratio from **0.32** to an efficient **0.78**.

## Installation & Usage
1. Clone the repository:
   ```bash
git clone [https://github.com/Akshat-Singh-Kshatriya/equity-esg-portfolio-optimizer.git](https://github.com/Akshat-Singh-Kshatriya/equity-esg-portfolio-optimizer.git)
cd equity-esg-portfolio-optimizer
 ```
2. Install the Dependencies
   ```bash
pip install -r requirements.txt
```
3. Run the Model
   ```bash
python equity_esg_portfolio.py
```
