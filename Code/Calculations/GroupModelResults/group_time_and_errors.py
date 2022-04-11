"""
AGRUPAR el tiempo, errores (MAE, …) en un solo excel (Indistinto del Index).

- Recoger el tiempo y errores de los archivos en Data/ModelsMetaResults/Time&Errors/{recurrence}/{
input}/time_and_errors_{timeframe}

Unir todos en un solo df:
| Recurrence | Input | Timeframe | TimeTaken in Minutes |TimeTaken in Hours | MAPE | MAE | RSME |

Guardar df en excel en el path: /Data/ModelsMetaResults/grouped_meta_results.xlsx

"""

from Calculations import constants
from Calculations import calculations
import pandas as pd

all_time_and_errors_dfs_list = []
for recurrence in constants.recurrences:
    for input_ in constants.inputs:
        for timeframe in constants.timeframes_dict.keys():
            path = f"""/Users/anuarnavarro/Desktop/TFG/GitHub/ForecastStockPrices/Code/Data/ModelsMetaResults/Time&Errors/{recurrence}/{input_}/time_and_errors_{timeframe}.xlsx"""
            time_and_errors_df = pd.read_excel(path, index_col=0)
            time_and_errors_df.insert(0, 'Timeframe', timeframe)
            time_and_errors_df.insert(0, 'Input', input_)
            time_and_errors_df.insert(0, 'Recurrence', recurrence)
            all_time_and_errors_dfs_list.append(time_and_errors_df)

all_time_and_errors_df = pd.concat(all_time_and_errors_dfs_list)
all_time_and_errors_df.to_excel(f"""/Users/anuarnavarro/Desktop/TFG/GitHub/ForecastStockPrices/Code/Data/grouped_meta_results.xlsx""")
