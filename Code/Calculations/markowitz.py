import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_prices(given_index, start_date, end_date):
    """
    get_prices returns pandas dataframe including prices between [start, end) dates for the stocks in specified index

    :param String given_index: index name
    :param String start_date: Date in YYYY-MM-DD format
    :param String end_date: Date in YYYY-MM-DD format
    :return: pandas dataframe including prices between start and end date for the stocks in specified index
    """
    if given_index == "DAX":
        absolute_path = \
            "/Users/anuarnavarro/Desktop/TFG/GitHub/ForecastStockPrices/Code/Data/DAX30/DAX30_cleaned_prices.xlsx"
    elif given_index == "S&P":
        absolute_path = \
            "/Users/anuarnavarro/Desktop/TFG/GitHub/ForecastStockPrices/Code/Data/S&P500/S&P500_cleaned_prices.xlsx"
    elif given_index == "DJI":
        absolute_path = \
            "/Users/anuarnavarro/Desktop/TFG/GitHub/ForecastStockPrices/Code/Data/DJI/DJI_cleaned_prices.xlsx"
    else:
        raise ValueError("Index name is not valid")

    df = pd.read_excel(absolute_path, index_col=0)
    df = df.loc[start_date: end_date]

    return df


def get_portfolios(given_index, start_date, end_date, window_number):
    """
    get_portfolios saves 2 portfolios for each window of 2 years:
    1.Using the sharpe ratio and 2.Markowitz min volatility portfolio.

    :param String given_index: index name
    :param String start_date: Date in YYYY-MM-DD format
    :param String end_date: Date in YYYY-MM-DD format
    :param Integer window_number: Starting from zero, the current window number (0 means 2005-2007, 1 means 2006-2007, ...)
    :return: N/A
    """
    index_prices_df = get_prices(given_index, start_date, end_date)
    # Calculate Covariance Matrix
    cov_matrix = index_prices_df.pct_change().apply(lambda x: np.log(1 + x)).cov()

    # NOT NEEDED - Calculate Portfolio variance given a list of Weights 'w' (risk||volatility of the portfolio)
    # port_var = cov_matrix.mul(w, axis=0).mul(w, axis=1).sum().sum()

    # Yearly returns for individual companies
    ind_er = index_prices_df.resample('Y').last().pct_change().mean()

    # NOT NEEDED - (Volatility || Risk) is given by the annual standard deviation. We multiply by 250 because
    # there are 250 trading days/year.
    # ann_sd = index_prices_df.pct_change().apply(lambda x: np.log(1 + x)).std().apply(lambda x: x * np.sqrt(250))

    # NOT NEEDED - Creating a table for visualising returns and volatility of assets
    # assets = pd.concat([ind_er, ann_sd], axis=1)
    # assets.columns = ['Returns', 'Volatility']

    # Next, to plot the graph of efficient frontier and calculate the variance of EACH PORTFOLIO, to the calculate the
    # Markowitz min volatility portfolio & Sharpe, we need to run a loop.
    # In each iteration, the loop considers different weights for assets and
    # calculates the return and volatility of that particular portfolio combination.
    # We run this loop a 10000 times.
    p_ret = []  # Define an empty array for portfolio returns
    p_vol = []  # Define an empty array for portfolio volatility
    p_weights = []  # Define an empty array for asset weights

    num_assets = len(index_prices_df.columns)
    num_portfolios = 10000
    for portfolio in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights = weights / np.sum(weights)
        p_weights.append(weights)
        returns = np.dot(weights,
                         ind_er)  # Returns are the product of individual expected returns of asset and its weights
        p_ret.append(returns)
        var = cov_matrix.mul(weights, axis=0).mul(weights, axis=1).sum().sum()  # Portfolio Variance
        sd = np.sqrt(var)  # Daily standard deviation
        ann_sd = sd * np.sqrt(252)  # Annual standard deviation = volatility
        p_vol.append(ann_sd)

    # Set data in df
    data = {'Returns': p_ret, 'Volatility': p_vol}

    for counter, symbol in enumerate(index_prices_df.columns.tolist()):
        data[symbol + ' weight'] = [w[counter] for w in p_weights]

    portfolios = pd.DataFrame(data)  # Dataframe of the 10000 portfolios created

    # The point (portfolios) in the interior are sub-optimal for a given risk level. For every interior point, there is
    # another that offers higher returns for the same risk.
    # On this graph, you can also see the combination of weights that will give you all possible combinations:
    # 1. Minimum volatility (left most point)
    # 2. Maximum returns (top most point)
    min_vol_port = portfolios.iloc[portfolios['Volatility'].idxmin()]

    # Sharpe Ratio -> An optimal risky portfolio can be considered as one that has highest Sharpe ratio.
    # Finding the optimal portfolio with the given risk factor based on 20 year bonds interests offered
    if given_index == "DAX":
        risk_factor = GERM_20y_bonds_avg_3_years[window_number]
    else:
        risk_factor = US_20y_bonds_avg_3_years[window_number]

    optimal_risky_port = portfolios.iloc[((portfolios['Returns'] - risk_factor) / portfolios['Volatility']).idxmax()]

    print(optimal_risky_port)  # Sharpe optimal portfolio
    print(min_vol_port)  # Min volatility (Markowitz) portfolio
    print("aqui")

    plot_markowitz_and_sharpe_and_efficient_frontier(portfolios, optimal_risky_port, min_vol_port)


def plot_markowitz_and_sharpe_and_efficient_frontier(portfolios, optimal_risky_port, min_vol_port):
    """
    plot_markowitz_and_sharpe_and_efficient_frontier

    :param String portfolios: all (about 10,000) possible portfolios
    :param Dataframe optimal_risky_port: Sharpe optimal portfolio in a dataframe
    :param Dataframe min_vol_port: Markowitz minimum volatility portfolio in a dataframe
    :return: N/A
    """
    # Plotting optimal portfolios (Min volatility + Sharpe)
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


US_20y_bonds_avg_3_years = [0.0485, 0.0475, 0.0446, 0.0418, 0.0395, 0.0343, 0.0311, 0.0292, 0.0292, 0.0262, 0.0247, 0.0262, 0.0269, 0.0227]
GERM_20y_bonds_avg_3_years = [0.0407, 0.0432, 0.0433, 0.0398, 0.0357, 0.0296, 0.0263, 0.0213, 0.0174, 0.0115, 0.0083, 0.0075, 0.0058, 0.0021]
indexes = ["DAX", "S&P", "DJI"]
dates = ["2005-01-01", "2006-01-01", "2007-01-01", "2008-01-01", "2009-01-01", "2010-01-01", "2011-01-01",
         "2012-01-01", "2013-01-01", "2014-01-01", "2015-01-01", "2016-01-01", "2017-01-01", "2018-01-01",
         "2019-01-01", "2020-01-01", "2021-01-01"]

years_window_size = 3
for index in indexes:
    for i in range(years_window_size, len(dates)):
        start_d = dates[i - years_window_size]
        end_d = dates[i]
        get_portfolios(index, "2009-01-01", "2019-01-01", i - years_window_size)
        raise ValueError("")
