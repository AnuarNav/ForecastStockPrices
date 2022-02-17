"""Gets the efficient portfolios for each 2 year window from [2005...2020) PREVIOUS to the next {trimester,
quarter, semester or annual} using Markowitz and Sharpe ratio based on previous prices for all indexes (DJI, DAX,
S&P) AND the return of each portfolio.

For each index and timeframe it saves a new excel file into
/Users/anuarnavarro/Desktop/TFG/GitHub/ForecastStockPrices/Code/Data/INDEX_NAME/Efficient Portfolios/
/INDEX_NAME_efficient_portfolios_and_returns_{timeframe}.xlsx

Result format (2 rows for each window, Markowitz and Sharpe):
| Strategy (Markowitz||Sharpe) | start_date | end_date | Volatility | Stock1 weight | Stock2 weight | ...  | Return
"""

from Calculations import constants
from Calculations import calculations
import pandas as pd


for timeframe in constants.timeframes_dict.keys():
    months = constants.timeframes_dict[timeframe]['months']
    dates = constants.timeframes_dict[timeframe]['dates']
    window_size = constants.timeframes_dict[timeframe]['window_size']
    for index in constants.indexes:
        opt_ports = []
        for i in range(window_size, len(dates)):
            start_d = dates[i - window_size]
            end_d = dates[i]
            opt_port_with_returns_df = calculations.get_portfolios(index, False, start_d, end_d,
                                                                   int((i - window_size) / window_size), months)
            opt_ports.append(opt_port_with_returns_df)

        opt_ports_df = pd.concat(opt_ports).reset_index(drop=True)

        opt_ports_df.to_excel(f'''/Users/anuarnavarro/Desktop/TFG/GitHub/ForecastStockPrices/Code/Data/{index}/Efficient Portfolios/{index}_efficient_portfolios_and_returns_{timeframe}.xlsx''')
        print(f'''################## Index {index} DONE ##################\n\n''')
    print(f'''################## Timeframe {timeframe} FINISHED ##################\n\n\n\n''')
