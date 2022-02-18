"""
This module gathers all functions to calculate efficient portfolios using markowitz and Sharpe ratio, as well as the
return of a portfolio comparing each component's closing price at end_date to closing price in one year time.
"""
import math

from Calculations import constants
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
from dateutil.relativedelta import relativedelta


def get_x_months_later_date(date_given, months):
    """
    Returns the date(date_given + months) in String format.

    :param date_given: Date in YYYY-MM-DD format
    :param months: Number of months wanted added to given_date
    :return: date(date_given + months) in String format
    """
    year = date_given[:4]
    month = date_given[5:7]
    next_date = date(int(year), int(month), 1) + relativedelta(months=+months)
    return str(next_date)


def get_prices(given_index, with_predicted, start_date, end_date, months_ahead=0, input_='', recurrence='',
               timeframe_name=''):
    """
    get_prices returns pandas dataframe including prices between [start, end) dates for the stocks in specified
    index. If with_predicted is true, it will also concatenate the predicted prices for next year into the dataframe

    :param timeframe_name: trimester/quarter/semester/annual
    :param recurrence: Recurrence of prediction prices wanted
    :param input_: Input of prediction prices wanted
    :param String given_index: index name
    :param Boolean with_predicted: True if the prices predicted should be used too
    :param String start_date: Date in YYYY-MM-DD format
    :param String end_date: Date in YYYY-MM-DD format
    :param Int months_ahead: Number of months prediction ahead
    :return: pandas dataframe including prices between start and end date for the stocks in specified index
    """
    absolute_path = f'''/Users/anuarnavarro/Desktop/TFG/GitHub/ForecastStockPrices/Code/Data/{given_index}/{given_index}_cleaned_prices.xlsx'''
    df = pd.read_excel(absolute_path, index_col=0)
    df = df.loc[start_date: end_date]

    if with_predicted:
        absolute_path = f"""/Users/anuarnavarro/Desktop/TFG/GitHub/ForecastStockPrices/Code/Data/{given_index}/PredictedPrices/{recurrence}/{input_}/{given_index}_predicted_prices_{timeframe_name}.xlsx"""

        # Get prices predicted starting from end_date (until X months ahead) and concat the predicted prices
        start_date = end_date
        end_date = get_x_months_later_date(end_date, months_ahead)
        df_predicted = pd.read_excel(absolute_path, index_col=0)
        df_predicted = df_predicted.loc[start_date: end_date]
        frames = [df, df_predicted]
        df = pd.concat(frames).groupby(level=0).last()

    return df


def get_only_predicted_prices(given_index, start_date, end_date, input_, recurrence, timeframe_name):
    """
    get_only_predicted_prices returns pandas dataframe including PREDICTED prices between [start, end) dates for the
    stocks in specified index, predcited by the specific recurrence+input+timeframe model.

    :param timeframe_name: trimester/quarter/semester/annual
    :param recurrence: Recurrence of prediction prices wanted
    :param input_: Input of prediction prices wanted
    :param String given_index: index name
    :param String start_date: Date in YYYY-MM-DD format
    :param String end_date: Date in YYYY-MM-DD format
    :return: pandas dataframe including prices between start and end date for the stocks in specified index
    """
    absolute_path = f"""/Users/anuarnavarro/Desktop/TFG/GitHub/ForecastStockPrices/Code/Data/{given_index}/PredictedPrices/{recurrence}/{input_}/{given_index}_predicted_prices_{timeframe_name}.xlsx"""

    # Get prices predicted from [start_date, end_date)
    df_predicted = pd.read_excel(absolute_path, index_col=0)
    df_predicted = df_predicted.loc[start_date: end_date]

    return df_predicted


def get_transposed_df(port_series):
    # Series to df + get rid of index
    port_df = pd.DataFrame(port_series).reset_index(level=0)

    port_df = port_df.T.iloc[:, 1:]  # Transpose df and delete Return column

    # Set column names to stock names (which curr are the first row)
    port_df = port_df.rename(columns=port_df.iloc[0]).iloc[1:, :]

    return port_df


def get_joint_ports_with_dates_and_strategy_df(optimal_risky_port_df, min_vol_port_df, end_date, months_ahead):
    frames = [min_vol_port_df, optimal_risky_port_df]
    joint_port_df = pd.concat(frames)

    # Add Strategy Column (first Markowitz, then Sharpe)
    strategies = ['Min volatility (Markowitz)', 'Sharpe Ratio']
    joint_port_df.insert(0, 'Strategy', strategies, True)
    # Add date columns and corresponding value (start and end are the dates of the 2 year window used to create
    # portfolio so, add X months to end_date and use orig end_date as start_date
    start_d = end_date
    end_date_portfolio = get_x_months_later_date(end_date, months_ahead)
    joint_port_df.insert(0, 'End Date', end_date_portfolio, True)
    joint_port_df.insert(0, 'Start Date', start_d, True)

    return joint_port_df


def get_portfolios(given_index, with_predicted, start_date, end_date, interval_2_year_index, months_ahead, input_='',
                   recurrence='', timeframe_name=''):
    """
    get_portfolios returns 2 portfolios for each window of 2 years:
    1.Using the sharpe ratio and 2.Markowitz min volatility portfolio.

    :param timeframe_name: trimester/quarter/semester/annual
    :param recurrence: Recurrence of prediction prices wanted
    :param input_: Input of prediction prices wanted
    :param String given_index: index name
    :param Boolean with_predicted: True if the prices predicted should be used too
    :param String start_date: Date in YYYY-MM-DD format
    :param String end_date: Date in YYYY-MM-DD format
    :param Integer interval_2_year_index: Starting from zero, the current window number (0 means 2005-2007, 1 means 2006-2007,..
    :param Int months_ahead: Number of months prediction ahead
    :return: df in the form | Strategy (Markowitz||Sharpe) | start_date | end_date | Stock1 weight | Stock2 weight |
    ... | Volatility | Return | -------- (2 rows for each window, Markowitz and Sharpe)
    """

    index_prices_df = get_prices(given_index, with_predicted, start_date, end_date, months_ahead=months_ahead)

    # Calculate Covariance Matrix
    cov_matrix = index_prices_df.pct_change().apply(lambda x: np.log(1 + x)).cov()

    # Yearly returns for individual companies (expected return)
    # Set data yearly by getting the last() value of each year, calculate the % change con respecto al año anterior and
    # calculate the mean of them
    ind_er = index_prices_df.resample('Y').last().pct_change().mean()

    # Next, to plot the graph of efficient frontier and calculate the variance of EACH PORTFOLIO, to the calculate the
    # Markowitz min volatility portfolio & Sharpe, we need to run a loop.
    # In each iteration, the loop considers different weights for assets and
    # calculates the return and volatility of that particular portfolio combination.
    # We run this loop a 10000 times.
    p_ret = []  # Define an empty array for portfolio returns
    p_vol = []  # Define an empty array for portfolio volatility
    p_weights = []  # Define an empty array for asset weights

    num_assets = len(index_prices_df.columns)
    for portfolio in range(constants.num_portfolios):
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
        data[symbol] = [w[counter] for w in p_weights]

    portfolios = pd.DataFrame(data)  # Dataframe of the 10000 portfolios created

    # The point (portfolios) in the interior are sub-optimal for a given risk level. For every interior point, there is
    # another that offers higher returns for the same risk.
    # On this graph, you can also see the combination of weights that will give you all possible combinations:
    # 1. Minimum volatility (left most point)
    # 2. Maximum returns (top most point)
    min_vol_port = portfolios.iloc[portfolios['Volatility'].idxmin()]

    # Sharpe Ratio -> An optimal risky portfolio can be considered as one that has highest Sharpe ratio.
    # Finding the optimal portfolio with the given risk factor based on 20 year bonds interests offered
    if given_index == "DAX30":
        # Apply mod to adapt to trimester/quarter/semester
        risk_factor = constants.GERM_20y_bonds_avg_2_years[interval_2_year_index]
    else:
        risk_factor = constants.US_20y_bonds_avg_2_years[interval_2_year_index]

    # Sharpe optimal portfolio
    optimal_risky_port = portfolios.iloc[((portfolios['Returns'] - risk_factor) / portfolios['Volatility']).idxmax()]

    optimal_risky_port_df = get_transposed_df(optimal_risky_port)
    min_vol_port_df = get_transposed_df(min_vol_port)

    min_vol_and_sharpe_joint_port_df = get_joint_ports_with_dates_and_strategy_df(
        optimal_risky_port_df, min_vol_port_df, end_date, months_ahead)

    opt_port_with_returns_df = calculate_returns(min_vol_and_sharpe_joint_port_df, given_index, end_date,
                                                 with_predicted, months_ahead)

    print(opt_port_with_returns_df)

    # plot_markowitz_and_sharpe_and_efficient_frontier(portfolios, optimal_risky_port, min_vol_port)

    return opt_port_with_returns_df


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


def calculate_returns(min_vol_and_sharpe_joint_port_df, given_index, end_date, with_predicted, months_ahead):
    """
    calculate_returns calculates the return of each portfolio in a df, using the formula:
    ROI = ∑(Wi * ln(Pfinal/Pinicial)) WHERE W = weight, Pfinal = closing price and Pinicial = starting price

    :param DataFrame min_vol_and_sharpe_joint_port_df: Pandas dataframe in the form | Strategy | start_date |
    end_date | Stock1 Weight | Stock2 Weight| ... |
     :param String given_index: index name
    :param String end_date: Date in YYYY-MM-DD format
    :param Boolean with_predicted: True if the prices predicted should be used too
    :return: df with a new column 'Return', representing the return from the specific portfolio compared to price in
    a trimester/quarter/semester/year tiem
    """

    start_date = end_date
    end_date = get_x_months_later_date(end_date, months_ahead)
    index_prices_df = get_prices(given_index, False, start_date, end_date)
    ports_returns = []
    for index, row in min_vol_and_sharpe_joint_port_df.iterrows():
        port_return = 0
        row = row[4:]  # Get rid of Dates & Strategy & Volatility
        for i in range(0, len(row)):
            stock_weight = row[i]
            start_price = index_prices_df.iloc[:, i].iloc[0]  # Select value in col i and first row
            end_price = index_prices_df.iloc[:, i].iloc[-1]  # Select value in col i and last row
            port_return += stock_weight * (math.log(end_price / start_price))

        ports_returns.append(port_return)

    returns_label = 'Return'
    if with_predicted:
        returns_label = 'Return_with_prediction'
        min_vol_and_sharpe_joint_port_df.rename(columns={'Volatility': 'Volatility_with_prediction'})

    min_vol_and_sharpe_joint_port_df[returns_label] = ports_returns

    return min_vol_and_sharpe_joint_port_df.reset_index(drop=True).groupby(level=0).last()
