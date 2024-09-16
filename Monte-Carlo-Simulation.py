import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from numba import jit, prange
import datetime as dt
import time

start_time = time.time()

file_path = 'port.xlsx'

# Load portfolio data efficiently
portfolio_data = pd.read_excel(file_path, sheet_name='Sheet1', usecols=['TICKER', 'ALLOCATION'], engine='openpyxl')
tickers = portfolio_data['TICKER'].tolist()
weights = portfolio_data['ALLOCATION'].values

# Load portfolio value
portfolio_value_df = pd.read_excel(file_path, sheet_name='Sheet1', usecols='D', nrows=1, engine='openpyxl')
portfolio_value = portfolio_value_df.iloc[0, 0]

# Download historical data once
data = yf.download(tickers, start='2014-01-01', end=dt.datetime.now())['Adj Close']

# Calculate daily returns
returns = data.pct_change().dropna()

# Simulation parameters
num_simulations = 100000
num_days = 252

# Mean returns and covariance matrix
mean_returns = returns.mean()
cov_matrix = returns.cov()

# Small regularization for numerical stability
cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-10

# Convert to NumPy arrays for compatibility with Numba
mean_returns_np = mean_returns.values  # Convert to NumPy array
cov_matrix_np = cov_matrix.values      # Convert to NumPy array

# Generate random daily returns outside the jitted function using NumPy's multivariate normal distribution
random_returns = np.random.multivariate_normal(mean_returns_np, cov_matrix_np, (num_simulations, num_days))

# Ensure weights is a NumPy array
weights = np.array(weights)

# Optimized Monte Carlo simulation using Numba
@jit(nopython=True, parallel=True)
def monte_carlo_sim(random_returns, weights, portfolio_value):
    num_days, num_simulations = random_returns.shape[1], random_returns.shape[0]
    simulation_results = np.zeros((num_days, num_simulations))
    portfolio_highs = np.zeros(num_simulations)
    portfolio_lows = np.zeros(num_simulations)

    for i in prange(num_simulations):
        cumulative_returns = np.cumprod(np.dot(random_returns[i], weights) + 1)
        portfolio_values = portfolio_value * cumulative_returns
        simulation_results[:, i] = portfolio_values
        portfolio_highs[i] = np.max(portfolio_values)
        portfolio_lows[i] = np.min(portfolio_values)

    return simulation_results, portfolio_highs, portfolio_lows

# Run the optimized simulation
simulation_results, portfolio_highs, portfolio_lows = monte_carlo_sim(random_returns, weights, portfolio_value)

# Calculate summary statistics
end_results = simulation_results[-1, :]
mean_ending_value = np.mean(end_results)
median_ending_value = np.median(end_results)
min_ending_value = np.min(end_results)
max_ending_value = np.max(end_results)
std_deviation = np.std(end_results)
correlation_matrix = returns.corr()

overall_high = np.max(portfolio_highs)
overall_low = np.min(portfolio_lows)

# Plot the simulation results
plt.figure(figsize=(10, 6))
plt.plot(simulation_results)
plt.title('Monte Carlo Simulation of Portfolio Returns')
plt.xlabel('Days')
plt.ylabel('Portfolio Value')

# Display the summary statistics on the plot
textstr = (f"Mean ending value: ${mean_ending_value:,.2f}\n"
           f"Median ending value: ${median_ending_value:,.2f}\n"
           f"Min ending value: ${min_ending_value:,.2f}\n"
           f"Max ending value: ${max_ending_value:,.2f}\n"
           f"Standard deviation: ${std_deviation:,.2f}\n"
           f"52-week high: ${overall_high:,.2f}\n"
           f"52-week low: ${overall_low:,.2f}")
plt.gcf().text(0.2, 0.75, textstr, fontsize=10, verticalalignment='center', bbox=dict(facecolor='white', alpha=0.5))

# Save the plot to a file
plt.savefig('portfolio_simulation_with_stats.png')

# Print the results
print(f"Mean ending portfolio value: ${mean_ending_value:,.2f}")
print(f"Median ending portfolio value: ${median_ending_value:,.2f}")
print(f"Minimum ending portfolio value: ${min_ending_value:,.2f}")
print(f"Maximum ending portfolio value: ${max_ending_value:,.2f}")
print(f"Standard deviation: ${std_deviation:,.2f}")
print(f"52-week high (across simulations): ${overall_high:,.2f}")
print(f"52-week low (across simulations): ${overall_low:,.2f}")

print("Correlation Matrix:")
print(correlation_matrix)

# Measure time taken for execution
end_time = time.time()
time_taken = end_time - start_time
print(f"Time taken: {time_taken:.2f} seconds")