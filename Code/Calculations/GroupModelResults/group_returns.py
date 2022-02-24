"""
AGRUPAR el return y return_with_prediction de cada modelo (con y sin predicción) para agrupar los returns de todos
los modelos en un solo archivo, facilitando así el script 'calculate_profit.py'

- Recoger el return O return_with_predicition de los archivos en Data/{Index}/Efficient Portfolios with Prediction/
{recurrence}/{input}/{Index}_efficient_portfolios_and_returns_with_prediction_{timeframe}.xlsx
Y de Data/{Index}/Efficient Portfolios/{Index}_efficient_portfolios_and_returns_{timeframe}.xlsx

Unir todos en un solo df:
| Index | With_prediction | Recurrence | Input | Timeframe | return

Guardar df en excel en el path: /Data/{index}/{index}_all_returns.xlsx

"""

from Calculations import constants
from Calculations import calculations
import pandas as pd

all_returns_dfs_list = []
for index in constants.indexes:
    for recurrence in constants.recurrences:
        for input_ in constants.inputs:
            for timeframe in constants.timeframes_dict.keys():
                path = f"""/Users/anuarnavarro/Desktop/TFG/GitHub/ForecastStockPrices/Code/Data/{index}/Efficient Portfolios with Prediction/{recurrence}/{input_}/{index}_efficient_portfolios_and_returns_with_prediction_{timeframe}.xlsx"""
                returns_df = pd.read_excel(path, index_col=0)
                port_return_mean = returns_df['Return_with_prediction'].mean()
                returns_df = pd.DataFrame({'Index': index, 'with_prediction': True,'Recurrence': recurrence,
                                           'Input': input_, 'Timeframe': timeframe,
                                           'return': port_return_mean},
                                          index=[0])
                all_returns_dfs_list.append(returns_df)

    for timeframe in constants.timeframes_dict.keys():
        path = f"""/Users/anuarnavarro/Desktop/TFG/GitHub/ForecastStockPrices/Code/Data/{index}/Efficient Portfolios/{index}_efficient_portfolios_and_returns_{timeframe}.xlsx"""
        returns_df = pd.read_excel(path, index_col=0)
        port_return_mean = returns_df['Return'].mean()
        returns_df = pd.DataFrame({'Index': index, 'with_prediction': False, 'Recurrence': '',
                                   'Input': '', 'Timeframe': timeframe,
                                   'return': port_return_mean},
                                  index=[0])
        all_returns_dfs_list.append(returns_df)

    all_compared_returns_df = pd.concat(all_returns_dfs_list)
    all_compared_returns_df.to_excel(f"""/Users/anuarnavarro/Desktop/TFG/GitHub/ForecastStockPrices/Code/Data/{index}/{index}_all_returns.xlsx""")
