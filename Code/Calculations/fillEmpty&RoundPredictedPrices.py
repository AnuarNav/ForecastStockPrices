"""Rounds predicted prices to 4 decimal places and, if any value is NaN, fills it with the original price
New file stored in {Index}/PredictedPricesCleaned/...
"""

from Calculations import constants
import pandas as pd

for recurrence in constants.recurrences:
    for input_ in constants.inputs:
        for timeframe in constants.timeframes_dict.keys():
            for index in constants.indexes:
                start_date = '2007-01-01'
                end_date = '2021-01-01'

                # Get original prices
                absolute_path = f'''/Users/anuarnavarro/Desktop/TFG/GitHub/ForecastStockPrices/Code/Data/{index}/{index}_cleaned_prices.xlsx'''
                df = pd.read_excel(absolute_path, index_col=0)
                df = df.loc[start_date: end_date]
                df = df.round(decimals=3)

                # Get prices predicted
                absolute_path = f"""/Users/anuarnavarro/Desktop/TFG/GitHub/ForecastStockPrices/Code/Data/{index}/PredictedPrices/{recurrence}/{input_}/{index}_predicted_prices_{timeframe}.xlsx"""
                df_predicted = pd.read_excel(absolute_path, index_col=0)
                df_predicted = df_predicted.loc[start_date: end_date]
                df_predicted = df_predicted.round(decimals=3)

                df_predicted = df_predicted.fillna(df)  # Fill NaN values in predicted prices with original prices

                for col in df_predicted:
                    indexes = df_predicted[(df_predicted[col] > 2000) | (df_predicted[col] < 0)].index.tolist()
                    for i in indexes:
                        df_predicted.loc[i, col] = df.loc[i, col]

                df_predicted.to_excel(
                    f'''/Users/anuarnavarro/Desktop/TFG/GitHub/ForecastStockPrices/Code/Data/{index}/PredictedPricesCleaned/{recurrence}/{input_}/{index}_predicted_prices_cleaned_{timeframe}.xlsx''')

                print(f''' ######################## Input->{input_} | Timeframe->{timeframe} | index->{index} ########################''')
            print(f''' ######################## Input->{input_} | Timeframe->{timeframe} ########################''')
        print(f''' ######################## Input->{input_} ########################''')
    print(f''' ######################## Recurrence->{recurrence} ########################''')
