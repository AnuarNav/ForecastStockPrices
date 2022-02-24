"""
AGRUPAR el mean_return_pct_change de cada modelo para poder comparar modelos entre sí

- Recoger el mean_return_pct_change de los archivos en Data/{Index}/Compared Returns/{recurrence}/
{input}/{Index}_returns_compared_{timeframe}.xlsx

Unir todos en un solo df:
| Index | Recurrence | Input | Timeframe | mean_pct_change

Guardar df en excel en el path: /Data/{index}/Compared Returns/grouped_compared_returns.xlsx

"""

from Calculations import constants
from Calculations import calculations
import pandas as pd

all_compared_returns_dfs_list = []
for index in constants.indexes:
    for recurrence in constants.recurrences:
        for input_ in constants.inputs:
            for timeframe in constants.timeframes_dict.keys():
                path = f"""/Users/anuarnavarro/Desktop/TFG/GitHub/ForecastStockPrices/Code/Data/{index}/Compared Returns/{recurrence}/{input_}/{index}_returns_compared_{timeframe}.xlsx"""
                compared_returns_df = pd.read_excel(path, index_col=0)
                mean_pct_change = compared_returns_df['mean_pct_change'].iloc[0]
                compared_returns_df = pd.DataFrame({'Index': index, 'Recurrence': recurrence, 'Input': input_,
                                                    'Timeframe': timeframe, 'mean_return_pct_change': mean_pct_change},
                                                   index=[0])
                all_compared_returns_dfs_list.append(compared_returns_df)

    all_compared_returns_df = pd.concat(all_compared_returns_dfs_list)
    all_compared_returns_df.to_excel(f"""/Users/anuarnavarro/Desktop/TFG/GitHub/ForecastStockPrices/Code/Data/{index}/Compared Returns/grouped_compared_returns.xlsx""")
