"""Gets the efficient portfolios for each 2 year window from [2005...2020) using Markowitz and Sharpe ratio based on
previous prices for all indexes (DJI, DAX, S&P) AND the return of each portfolio.

For each index it saves a new excel file into
/Users/anuarnavarro/Desktop/TFG/GitHub/ForecastStockPrices/Code/Data/INDEX_NAME
/INDEX_NAME_efficient_portfolios_and_returns.xlsx

Result format (2 rows for each window, Markowitz and Sharpe):
| Strategy (Markowitz||Sharpe) | start_date | end_date | Stock1 weight | Stock2 weight | ...  | Return
"""

import constants
import calculations


for index in constants.indexes:
    for i in range(constants.years_window_size, len(constants.dates)):
        start_d = constants.dates[i - constants.years_window_size]
        end_d = constants.dates[i]
        calculations.get_portfolios(index, False, start_d, end_d, i - constants.years_window_size)
        raise ValueError("")
