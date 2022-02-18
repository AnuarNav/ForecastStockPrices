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


def get_model_time_taken_df(recurrence_, input_given, timeframe_name):
    input_with_underscore = constants.inputs_with_underscore[input_given]
    timeframe_number = constants.timeframes_dict[timeframe_name]['timeframe_number']
    time_absolute_path = f"""/Users/anuarnavarro/Desktop/TFG/GitHub/ForecastStockPrices/Code/Data/ModelsMetaResults/Time/{recurrence_}/{input_given}/{input_with_underscore}_output_{timeframe_number}_time.xlsx"""
    time_df = pd.read_excel(time_absolute_path, index_col=0)

    return time_df


def get_orig_and_predicted_values(start_date, end_date, input__, recurrence_, timeframe_):
    """Returns the orig and predicted values for a specific model of all 3 indexes in 2 arrays, orig and predicted
    values"""
    orig_prices_dfs_list = []  # Lists to store all values predicted for EACH Index same model has predicted
    predicted_prices_dfs_list = []
    for index in constants.indexes:
        orig_prices_df = calculations.get_prices(
            index, False, start_date, end_date, input_=input__, recurrence=recurrence_, timeframe_name=timeframe_)
        predicted_prices_df = calculations.get_only_predicted_prices(
            index, start_date, end_date, input__, recurrence=recurrence_, timeframe_name=timeframe_)
        orig_prices_dfs_list.append(orig_prices_df)
        predicted_prices_dfs_list.append(predicted_prices_df)

    # Concat all 3 indexes orig/predictions into a single df
    all_orig_prices_df = pd.concat(orig_prices_dfs_list, axis=1).dropna()
    all_predicted_prices_df = pd.concat(predicted_prices_dfs_list, axis=1).dropna()

    orig_stacked_columns_df = all_orig_prices_df.stack().reset_index()
    predicted_columns_df = all_predicted_prices_df.stack().reset_index()

    merged_df = pd.merge(orig_stacked_columns_df, predicted_columns_df, on=['Date', 'level_1']).dropna()

    original_price_values = merged_df['0_x']
    predicted_price_values = merged_df['0_y']

    return original_price_values, predicted_price_values


start_d = '2007-01-04'
end_d = '2021-01-01'
for recurrence in constants.recurrences:
    for input_ in constants.inputs:
        for timeframe in constants.timeframes_dict.keys():
            orig_values, predicted_values = get_orig_and_predicted_values(start_d, end_d, input_, recurrence, timeframe)

            mape = sk.mean_absolute_percentage_error(orig_values, predicted_values)
            mae = sk.mean_absolute_error(orig_values, predicted_values)
            rsme = sk.mean_squared_error(orig_values, predicted_values, squared=False)

            time_taken_df = get_model_time_taken_df(recurrence, input_, timeframe)

            time_taken_df['MAPE'] = mape
            time_taken_df['MAE'] = mae
            time_taken_df['RSME'] = rsme

            time_taken_df.to_excel(f'''/Users/anuarnavarro/Desktop/TFG/GitHub/ForecastStockPrices/Code/Data/ModelsMetaResults/Time&Errors/{recurrence}/{input_}/time_and_errors_{timeframe}.xlsx''')
