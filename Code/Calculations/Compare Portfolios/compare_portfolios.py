"""This module compares the returns of each portfolio with it's time equivalent predicted portfolio, generating a new
file which contains the return_pct_change for each row, calculated by:
return_pct_change = ((Return_with_prediction - Return) / Return) * 100

Files Compared:
'INDEX_efficient_portfolios_and_returns_with_prediction.xlsx’ and ‘INDEX_efficient_portfolios_and_returns.xlsx’

For each index, input, timeframe and recurrence it saves a new excel file into:
/Users/anuarnavarro/Desktop/TFG/GitHub/ForecastStockPrices/Code/Data/INDEX_NAME/Compared Returns/{RECURRENCE}/{INPUT}
/INDEX_NAME_returns_compared_{TimeFrameWindow}.xlsx

Result format:
| Strategy (Markowitz/Sharpe) | Start_date | end_date | return | return_with_prediction | return_pct_change |
mean_return_pct_change|"""

from Calculations import constants
import pandas as pd


def get_returns_and_with_predicted(index_given, timeframe_, input_given, recurrence_):
    """
    Given the index name, returns one dataframe built by joining returns with and without using predicted prices

    :param timeframe_: trimester/quarter/semester/annual
    :param recurrence_: Recurrence of prediction prices wanted
    :param input_: Input of prediction prices wanted
    :param String index_given: index name
    :return: Returns one dataframe built by joining returns with and without using predicted prices in the form:
    | start_date | end_date | Return | Return_with_prediction |
    """
    absolute_path = f'''//Users/anuarnavarro/Desktop/TFG/GitHub/ForecastStockPrices/Code/Data/{index_given}/Efficient Portfolios/{index_given}_efficient_portfolios_and_returns_with_prediction_{timeframe_}.xlsx'''
    absolute_path_with_prediction = f'''//Users/anuarnavarro/Desktop/TFG/GitHub/ForecastStockPrices/Code/Data/{index_given}/Efficient Portfolios with Prediction/{recurrence_}/{input_given}/{index_given}_efficient_portfolios_and_returns_with_prediction_{timeframe_}.xlsx'''

    returns_df = pd.read_excel(absolute_path, index_col=0)
    returns_with_prediction_df = pd.read_excel(absolute_path_with_prediction, index_col=0)

    # Get column with returns_with_prediction from corresponding dataframe and join it with returns_df
    returns_with_prediction = returns_with_prediction_df['Return_with_prediction']
    returns_w_and_wo_prediction_df = returns_df.join(returns_with_prediction)

    # Drop all stock weight columns
    returns_w_and_wo_prediction_df.drop(returns_w_and_wo_prediction_df.columns.
                                        difference(['Strategy', 'Start Date', 'End Date', 'Return',
                                                    'Return_with_prediction', 'Volatility',
                                                    'Volatility_with_prediction']), 1, inplace=True)

    return returns_w_and_wo_prediction_df


def get_return_pct_change_and_mean(returns_w_and_wo_prediction_df):
    """

    :param returns_w_and_wo_prediction_df: df in the form | start_date | end_date | Return | Return_with_prediction |
    :return: Same df as given BUT adding two new columns: return_pct_change, calculated with the formula:
    return_pct_change = ((Return_with_prediction - Return) / Return) * 100
    and mean_return_pct_change, being the mean of all return_pct_change values
    Result in format:
    | start_date | end_date | Volatility | Volatility_with_prediction | Return | Return_with_prediction |
    return_pct_change | mean_return_pct_change |
    """

    returns_w_and_wo_prediction_df['return_pct_change'] = returns_w_and_wo_prediction_df.apply(
        lambda row: ((row.Return_with_prediction - row.Return) / row.Return) * 100, axis=1
    )

    # Set mean in first row in new column 'mean_return_pct_change'
    returns_w_and_wo_prediction_df.loc[returns_w_and_wo_prediction_df.index[0], 'mean_return_pct_change'] = \
        returns_w_and_wo_prediction_df['return_pct_change'].mean().item()

    return returns_w_and_wo_prediction_df


for recurrence in constants.recurrences:
    for input_ in constants.inputs:
        for timeframe in constants.timeframes_dict.keys():
            for index in constants.indexes:
                returns_with_and_without_prediction_df = get_returns_and_with_predicted(index_given=index,
                                                                                        timeframe_=timeframe,
                                                                                        input_given=input_,
                                                                                        recurrence_=recurrence)
                returns_with_return_pct_change_df = get_return_pct_change_and_mean(returns_with_and_without_prediction_df)

                returns_with_return_pct_change_df.to_excel(
                    f'''/Users/anuarnavarro/Desktop/TFG/GitHub/ForecastStockPrices/Code/Data/{index}/Compared Returns/{recurrence}/{input_}//{index}_returns_compared_{timeframe}.xlsx''')
