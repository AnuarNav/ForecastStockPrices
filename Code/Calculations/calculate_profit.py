"""
Recogiendo el return de cada índice, guardado en
/Data/{index}/{index}_all_returns.xlsx ...

... Este script calcula la rentabilidad total, coste total y beneficio para cada tipo de cartera (trimestral,
cuatrimestral, semestral y anual Y con predicción y sin predicción Y para cada ÍNDICE)

guardando los resultados de cada índice en el siguiente formato:
| Index | With_prediction | Recurrence | Input | Timeframe | Return | Revenue | Total Cost | Profit |

Where:
Revenue = initial_investment * return
Total cost = creation_cost + annual_maintenance_cost * 14 + transaction_fee * initial_investment * periods * 2(buy/sell)
Profit = Revenue - Total Cost

en el archivo /Data/{index}/{index}_all_returns_and_monetary_results.xlsx
"""
import pandas

from Calculations import constants
import pandas as pd

path = f"""/Users/anuarnavarro/Desktop/TFG/GitHub/ForecastStockPrices/Code/Data/S&P500/S&P500_all_returns.xlsx"""
returns_df = pd.read_excel(path, index_col=0)
returns_df['Revenue'] = returns_df.Return * constants.initial_investment
returns_df['Total_cost'] = returns_df.Timeframe.apply(
    lambda tf: constants.creation_cost + constants.annual_maintenance_cost * 14 +
    constants.transaction_fee * constants.initial_investment * constants.timeframes_dict[tf]['periods'] * 2)
returns_df['profit'] = returns_df.Revenue * returns_df.Total_cost

returns_df.to_excel(
    f"""/Users/anuarnavarro/Desktop/TFG/GitHub/ForecastStockPrices/Code/Data/all_returns_and_monetary_results.xlsx""")
