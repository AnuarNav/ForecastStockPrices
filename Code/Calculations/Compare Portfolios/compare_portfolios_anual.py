"""This module compares the returns of each portfolio with it's time equivalent predicted portfolio, generating a new
file which contains the pct_change for each row, calculated by:
pct_change = ((Return_with_prediction - Return) / Return) * 100

Files Compared:
'INDEX_efficient_portfolios_and_returns_with_prediction.xlsx’ and ‘INDEX_efficient_portfolios_and_returns.xlsx’

For each index it saves a new excel file into:
/Users/anuarnavarro/Desktop/TFG/GitHub/ForecastStockPrices/Code/Data/INDEX_NAME/Compared Returns/{RECURRENCE}/{INPUT}
/INDEX_NAME_returns_compared_{TimeFrameWindow}.xlsx

Result format:
| Strategy (Markowitz/Sharpe) | Start_date | end_date | return | return_with_prediction | pct_change |
mean_pct_change|"""

from Calculations import constants
import pandas as pd

"""    - [ ] PONER LAS VARS Q CAMBIAN ARRIBA y SOLO CAMBIAR ESO EN CADA SCRIPT"""

def get_returns_and_with_predicted(index_given):
    """
    Given the index name, returns one dataframe built by joining returns with and without using predicted prices

    :param String index_given: index name
    :return: Returns one dataframe built by joining returns with and without using predicted prices in the form:
    | start_date | end_date | Return | Return_with_prediction |
    """
    if index_given == "DAX30":
        absolute_path = \
            "/Data/DAX30/DAX30_efficient_portfolios_and_returns_anual.xlsx"
        absolute_path_with_prediction = \
            "/Data/DAX30/DAX30_efficient_portfolios_and_returns_with_prediction_anual_auto.xlsx"
    elif index_given == "S&P500":
        absolute_path = \
            "/Data/S&P500/S&P500_efficient_portfolios_and_returns.xlsx"
        absolute_path_with_prediction = \
            "/Users/anuarnavarro/Desktop/TFG/GitHub/ForecastStockPrices/Code/Data/S&P500/S&P500_efficient_portfolios_and_returns_with_prediction.xlsx"
    elif index_given == "DJI":
        absolute_path = \
            "/Data/DJI/DJI_efficient_portfolios_and_returns.xlsx"
        absolute_path_with_prediction = \
            "/Users/anuarnavarro/Desktop/TFG/GitHub/ForecastStockPrices/Code/Data/DJI/DJI_efficient_portfolios_and_returns_with_prediction.xlsx"
    else:
        raise ValueError("Index name is not valid")

    returns_df = pd.read_excel(absolute_path, index_col=0)
    returns_with_prediction_df = pd.read_excel(absolute_path_with_prediction, index_col=0)

    # Get column with returns_with_prediction from corresponding dataframe and join it with returns_df
    returns_with_prediction = returns_with_prediction_df['Return_with_prediction']
    returns_w_and_wo_prediction_df = returns_df.join(returns_with_prediction)

    # Drop all stock weight columns
    returns_w_and_wo_prediction_df.drop(returns_w_and_wo_prediction_df.columns.
                                        difference(['Strategy', 'Start Date', 'End Date', 'Return',
                                                    'Return_with_prediction']), 1, inplace=True)

    return returns_w_and_wo_prediction_df


def get_pct_change_and_mean(returns_w_and_wo_prediction_df):
    """

    :param returns_w_and_wo_prediction_df: df in the form | start_date | end_date | Return | Return_with_prediction |
    :return: Same df as given BUT adding two new columns: pct_change, calculated with the formula:
    pct_change = ((Return_with_prediction - Return) / Return) * 100
    and mean_pct_change, being the mean of all pct_change values
    Result in format:
    | start_date | end_date | Return | Return_with_prediction | pct_change | mean_pct_change
    """

    returns_w_and_wo_prediction_df['pct_change'] = returns_w_and_wo_prediction_df.apply(
        lambda row: ((row.Return_with_prediction - row.Return) / row.Return) * 100, axis=1
    )

    # Set mean in first row in new column 'mean_pct_change'
    returns_w_and_wo_prediction_df.loc[returns_w_and_wo_prediction_df.index[0], 'mean_pct_change'] = \
        returns_w_and_wo_prediction_df['pct_change'].mean().item()

    return returns_w_and_wo_prediction_df

for recurrence in constants.recurrences:
    for input in constants.inputs:
        for index in constants.indexes:
            returns_with_and_without_prediction_df = get_returns_and_with_predicted(index)
            returns_with_pct_change_df = get_pct_change_and_mean(returns_with_and_without_prediction_df)

            returns_with_pct_change_df.to_excel(f'''/Users/anuarnavarro/Desktop/TFG/GitHub/ForecastStockPrices/Code/Data/{index}/{index}_returns_compared.xlsx''')
            raise ValueError("")
