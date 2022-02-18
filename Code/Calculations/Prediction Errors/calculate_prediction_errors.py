"""
For every model created (recurrence+input+output), gets the ModelsMetaResults/Time into a df (even if empty) and stores
values calculates for the given recurrence+input+output the following errors: MAPE, MAE, RSME

Saves resulting df into:
/Data/ModelsMetaResults/Time&Errors/{recurrence}/{input}/{time_file_name}_errors.xlsx
"""


from Calculations import constants
from Calculations import calculations
import pandas as pd
import sklearn.metrics as sk


def get_model_time_taken_df(given_index, recurrence_, input_given, timeframe_name):
    input_with_underscore = constants.inputs_with_underscore[input_given]
    timeframe_number = constants.timeframes_dict[timeframe_name]['timeframe_number']
    time_absolute_path = f"""/Users/anuarnavarro/Desktop/TFG/GitHub/ForecastStockPrices/Code/Data/ModelsMetaResults/Time/{recurrence_}/{input_given}/{input_with_underscore}_output_{timeframe_number}_time.xlsx"""
    time_df = pd.read_excel(time_absolute_path, index_col=0)

    return time_df


for recurrence in constants.recurrences:
    for input_ in constants.inputs:
        for timeframe in constants.timeframes_dict.keys():
            months = constants.timeframes_dict[timeframe]['months']
            dates = constants.timeframes_dict[timeframe]['dates']
            window_size = constants.timeframes_dict[timeframe]['window_size']
            for index in constants.indexes:
                start_d = '2007-01-04'
                end_d = '2021-01-01'
                orig_prices_df = calculations.get_prices(
                    index, False, start_d, end_d, input_=input_, recurrence=recurrence, timeframe_name=timeframe)
                predicted_prices_df = calculations.get_only_predicted_prices(
                    index, start_d, end_d, input_, recurrence=recurrence, timeframe_name=timeframe)
                orig_stacked_columns_df = orig_prices_df.stack().reset_index()
                predicted_columns_df = predicted_prices_df.stack().reset_index()
                merged_df = pd.merge(orig_stacked_columns_df, predicted_columns_df, on=['Date', 'level_1']).dropna()
                orig_values = merged_df['0_x']
                predicted_values = merged_df['0_y']

                mape = sk.mean_absolute_percentage_error(orig_values, predicted_values)
                mae = sk.mean_absolute_error(orig_values, predicted_values)
                rsme = sk.mean_squared_error(orig_values, predicted_values, squared=False)

                time_taken_df = get_model_time_taken_df(index, recurrence, input_, timeframe)

                time_taken_df['MAPE'] = mape
                time_taken_df['MAE'] = mae
                time_taken_df['RSME'] = rsme

                time_taken_df.to_excel(f'''/Users/anuarnavarro/Desktop/TFG/GitHub/ForecastStockPrices/Code/Data/ModelsMetaResults/Time&Errors/{recurrence}/{input_}/{index}_time_and_errors_{timeframe}.xlsx''')
