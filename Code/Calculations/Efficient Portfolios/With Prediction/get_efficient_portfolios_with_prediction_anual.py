"""Gets the efficient portfolios for each {year} based on each 2 year window previous to {year} + next {
year} of predicted prices (results of  prediction saved in
/Users/anuarnavarro/Desktop/TFG/GitHub/ForecastStockPrices/Code/Data/INDEX_NAME/PredictedPrices/{RECURRENCE}/{INPUT}/
INDEX_NAME_predicted_prices_output_{NUMBER_OF_PRICES}.xlsx.) using Markowitz and Sharpe ratio based on previous prices
for all indexes (DJI,DAX, S&P) AND calculates the return of each portfolio. Portfolio for next QUARTER.

For each index AND each INPUT AND AUTO/MANUAL Recurrence size in prediction it saves a new excel file into:
/Users/anuarnavarro/Desktop/TFG/GitHub/ForecastStockPrices/Code/Data/INDEX_NAME/Efficient Portfolios with
Prediction/{INPUT}/INDEX_NAME_efficient_portfolios_and_returns_with_prediction_{year}.xlsx

Result format (2 rows for each window, Markowitz and Sharpe):
| Strategy (Markowitz||Sharpe) | start_date | end_date | Stock1 weight | Stock2 weight | ...  | Return
"""

from Calculations import constants
from Calculations import calculations
import pandas as pd

"""    - [ ] PONER LAS VARS Q CAMBIAN ARRIBA y SOLO CAMBIAR ESO EN CADA SCRIPT"""

for recurrence in constants.recurrences:
    for input in constants.inputs:
        for index in constants.indexes:
            opt_ports = []
            for i in range(constants.years_window_size, len(constants.annual_dates)):
                start_d = constants.annual_dates[i - constants.years_window_size]
                end_d = constants.annual_dates[i]
                opt_port_with_returns_df = calculations.get_portfolios(index, True, start_d, end_d,
                                                                       i - constants.years_window_size)
                opt_ports.append(opt_port_with_returns_df)

            opt_ports_df = pd.concat(opt_ports).reset_index(drop=True)

            opt_ports_df.to_excel(f'''/Users/anuarnavarro/Desktop/TFG/GitHub/ForecastStockPrices/Code/Data/{index}/{index}_efficient_portfolios_and_returns_with_prediction.xlsx''')
