import pandas as pd
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

table=pd.read_html('https://en.wikipedia.org/wiki/DAX')

df = table[3]

df.to_csv("/Users/anuarnavarro/Desktop/TFG/GitHub/ForecastStockPrices/Code/Data/DAX30/DAX30-Symbols.csv", columns=['Ticker symbol'])