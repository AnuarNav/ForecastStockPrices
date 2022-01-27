import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

US_20y_bonds_avg_2_years = [4.82, 4.95, 4.63, 4.24, 4.08, 3.88, 3.12, 2.82, 3.10, 2.83, 2.38, 2.43, 2.84, 2.71]
GERM_10y_bonds_avg_2_years = [3.88, 4.24, 4.46, 4.27, 3.73, 3.33, 2.75, 2.30, 2.08, 1.44, 0.79, 0.74, 0.84, 0.43]

def get_prices(absolute_path):
    df = pd.read_excel(absolute_path, index_col=0)
    return df


DAX30_df = get_prices("/Users/anuarnavarro/Desktop/TFG/Code/Data/DAX30/DAX30_cleaned_prices.xlsx")

# Calculate Covariance Matrix
cov_matrix = DAX30_df.pct_change().apply(lambda x: np.log(1 + x)).cov()

# NOT NEEDED - Calculate Portfolio variance given a list of Weights 'w' (risk||volatility of the portfolio)
# port_var = cov_matrix.mul(w, axis=0).mul(w, axis=1).sum().sum()

# Yearly returns for individual companies
ind_er = DAX30_df.resample('Y').last().pct_change().mean()

# NOT NEEDED - (Volatility || Risk) is given by the annual standard deviation. We multiply by 250 because
# there are 250 trading days/year.
ann_sd = DAX30_df.pct_change().apply(lambda x: np.log(1+x)).std().apply(lambda x: x*np.sqrt(250))

# NOT NEEDED - Creating a table for visualising returns and volatility of assets
assets = pd.concat([ind_er, ann_sd], axis=1)
assets.columns = ['Returns', 'Volatility']

# Next, to plot the graph of efficient frontier, we need run a loop.
# In each iteration, the loop considers different weights for assets and
# calculates the return and volatility of that particular portfolio combination.
# We run this loop a 1000 times.
p_ret = []  # Define an empty array for portfolio returns
p_vol = []  # Define an empty array for portfolio volatility
p_weights = []  # Define an empty array for asset weights

num_assets = len(DAX30_df.columns)
num_portfolios = 10000
for portfolio in range(num_portfolios):
    weights = np.random.random(num_assets)
    weights = weights/np.sum(weights)
    p_weights.append(weights)
    returns = np.dot(weights, ind_er)  # Returns are the product of individual expected returns of asset and its weights
    p_ret.append(returns)
    var = cov_matrix.mul(weights, axis=0).mul(weights, axis=1).sum().sum()  # Portfolio Variance
    sd = np.sqrt(var)  # Daily standard deviation
    ann_sd = sd*np.sqrt(250)  # Annual standard deviation = volatility
    p_vol.append(ann_sd)

# Set data in df
data = {'Returns': p_ret, 'Volatility': p_vol}

for counter, symbol in enumerate(DAX30_df.columns.tolist()):
    data[symbol+' weight'] = [w[counter] for w in p_weights]

portfolios = pd.DataFrame(data)
portfolios.head()  # Dataframe of the 10000 portfolios created

# Plot efficient frontier
"""portfolios.plot.scatter(x='Volatility', y='Returns', marker='o', s=10, alpha=0.3, grid=True, figsize=[10, 10])"""

# The point (portfolios) in the interior are sub-optimal for a given risk level. For every interior point, there is
# another that offers higher returns for the same risk.
# On this graph, you can also see the combination of weights that will give you all possible combinations:
# 1. Minimum volatility (left most point)
# 2. Maximum returns (top most point)
min_vol_port = portfolios.iloc[portfolios['Volatility'].idxmin()]
print(min_vol_port)
# Plot min vol portfolio
"""plt.subplots(figsize=[10, 10])
plt.scatter(portfolios['Volatility'], portfolios['Returns'], marker='o', s=10, alpha=0.3)
plt.scatter(min_vol_port[1], min_vol_port[0], color='r', marker='*', s=500)"""

# Sharpe Ratio -> An optimal risky portfolio can be considered as one that has highest Sharpe ratio.
# Finding the optimal portfolio with
rf = 0.01  # risk factor
optimal_risky_port = portfolios.iloc[((portfolios['Returns']-rf)/portfolios['Volatility']).idxmax()]
print(optimal_risky_port)
# Plotting optimal portfolio
portfolios.plot.scatter(x='Volatility', y='Returns', marker='o', s=10, alpha=0.3, grid=True, figsize=[10, 10])

plt.scatter(min_vol_port[1], min_vol_port[0], color='r', marker='*', s=200)
# label min volatility point
plt.annotate("(" + str(round(min_vol_port[1], 3)) + " " + str(round(min_vol_port[0], 3)) + ")",
             (min_vol_port[1], min_vol_port[0]))

plt.scatter(optimal_risky_port[1], optimal_risky_port[0], color='g', marker='*', s=200)
# label optimal point (Sharpe point)
plt.annotate("(" + str(round(optimal_risky_port[1], 3)) + " " + str(round(optimal_risky_port[0], 3)) + ")",
             (optimal_risky_port[1], optimal_risky_port[0]))

plt.show()
