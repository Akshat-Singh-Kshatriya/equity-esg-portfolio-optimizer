import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize

tickers = ['AAPL', 'MSFT', 'PG', 'JNJ', 'NEE']
start_date = '2018-01-01'
end_date = '2024-01-01'

price_data = yf.download(tickers, start=start_date, end=end_date)['Close']

returns = price_data.pct_change().dropna()

trading_days = 252
risk_free_rate = 0.07

mean_returns = returns.mean() * trading_days
cov_matrix = returns.cov() * trading_days

def portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate):
    port_return = np.sum(mean_returns * weights)
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (port_return - risk_free_rate) / port_volatility
    return port_return, port_volatility, sharpe_ratio

def negative_sharpe(weights, mean_returns, cov_matrix, risk_free_rate):
    _, _, p_sharpe = portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)
    return -p_sharpe

num_assets = len(tickers)
args = (mean_returns, cov_matrix, risk_free_rate)
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  
bounds = tuple((0.05, 0.40) for _ in range(num_assets))              
initial_guess = num_assets * [1. / num_assets]

optimized_result = minimize(
    negative_sharpe, 
    initial_guess, 
    args=args,
    method='SLSQP', 
    bounds=bounds, 
    constraints=constraints
)

optimal_weights = optimized_result.x
opt_return, opt_volatility, opt_sharpe = portfolio_performance(
    optimal_weights, mean_returns, cov_matrix, risk_free_rate
)

for ticker, weight in zip(tickers, optimal_weights):
    print(f"{ticker:4}: {weight * 100:>5.2f}%")

print(f"Annualized Return:            {opt_return * 100:.2f}%")
print(f"Annualized Volatility (Risk): {opt_volatility * 100:.2f}%")
print(f"Sharpe Ratio:                 {opt_sharpe:.2f}")

sns.set_theme(style="whitegrid")
plt.figure(figsize=(8, 8))

display_weights = [w for w in optimal_weights if w > 0.001]
display_tickers = [t for t, w in zip(tickers, optimal_weights) if w > 0.001]
plt.pie(
    display_weights, 
    labels=display_tickers, 
    autopct='%1.1f%%', 
    startangle=140, 
    colors=sns.color_palette("pastel"),
    wedgeprops={'edgecolor': 'black'}
)
plt.title("Optimized Portfolio Allocation (Maximized Sharpe Ratio)", fontsize=16)
plt.tight_layout()
plt.show()