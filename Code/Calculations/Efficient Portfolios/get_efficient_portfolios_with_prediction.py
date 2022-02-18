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

for recurrence in constants.recurrences:
    for input_ in constants.inputs:
        for timeframe in constants.timeframes_dict.keys():
            months = constants.timeframes_dict[timeframe]['months']
            dates = constants.timeframes_dict[timeframe]['dates']
            window_size = constants.timeframes_dict[timeframe]['window_size']
            for index in constants.indexes:
                opt_ports = []
                for i in range(window_size, len(dates)):
                    start_d = dates[i - window_size]
                    end_d = dates[i]
                    opt_port_with_returns_df = calculations.get_portfolios(
                        index, True, start_d, end_d, int((i - window_size) / window_size), months_ahead=months,
                        timeframe_name=timeframe, input_=input_, recurrence=recurrence)
                    opt_ports.append(opt_port_with_returns_df)

                opt_ports_df = pd.concat(opt_ports).reset_index(drop=True)

                opt_ports_df.to_excel(f'''/Users/anuarnavarro/Desktop/TFG/GitHub/ForecastStockPrices/Code/Data/{index}/Efficient Portfolios with Prediction/{recurrence}/{input_}/{index}_efficient_portfolios_and_returns_with_prediction_{timeframe}.xlsx''')
                print(f'''################## Index {index} DONE ##################\n\n''')
            print(f'''################## Timeframe {timeframe} FINISHED ##################\n\n\n\n''')
        print(f'''################## Input {input_} COMPLETED ##################\n\n\n\n''')
    print(f'''################## Recurrence {recurrence} ENDED ##################\n\n\n\n''')
