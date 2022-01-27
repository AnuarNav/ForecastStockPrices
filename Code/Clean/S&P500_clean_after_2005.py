import pandas as pd

# create a Dataframe
df = pd.read_excel("/Users/anuarnavarro/Desktop/TFG/GitHub/ForecastStockPrices/Code/Data/S&P500/S&P500_after_2005_data.xlsx")

nan_value = float("NaN")
df.replace(0, nan_value, inplace=True)

df.dropna(axis=1, inplace=True, thresh=4028)

df.to_excel('/Users/anuarnavarro/Desktop/TFG/GitHub/ForecastStockPrices/Code/Data/S&P500/S&P500_cleaned_prices.xlsx')
