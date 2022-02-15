"""Gets the efficient portfolios for each 2 year window from [2005...2020) PREVIOUS to the next {semester} using
Markowitz and Sharpe ratio based on previous prices for all indexes (DJI, DAX, S&P) AND the return of each portfolio.

For each index it saves a new excel file into
/Users/anuarnavarro/Desktop/TFG/GitHub/ForecastStockPrices/Code/Data/INDEX_NAME
/INDEX_NAME_efficient_portfolios_and_returns.xlsx

Result format (2 rows for each window, Markowitz and Sharpe):
| Strategy (Markowitz||Sharpe) | start_date | end_date | Stock1 weight | Stock2 weight | ...  | Return
"""

from Calculations import constants
from Calculations import calculations
import pandas as pd


for index in constants.indexes:
    opt_ports = []
    for i in range(constants.annual_window_size, len(constants.annual_dates)):
        start_d = constants.annual_dates[i - constants.annual_window_size]
        end_d = constants.annual_dates[i]
        opt_port_with_returns_df = calculations.get_portfolios(index, False, start_d, end_d,
                                                               i - constants.annual_window_size)
        opt_ports.append(opt_port_with_returns_df)

    opt_ports_df = pd.concat(opt_ports).reset_index(drop=True)

    opt_ports_df.to_excel(f'''/Users/anuarnavarro/Desktop/TFG/GitHub/ForecastStockPrices/Code/Data/{index}/{index}_efficient_portfolios_and_returns.xlsx''')
