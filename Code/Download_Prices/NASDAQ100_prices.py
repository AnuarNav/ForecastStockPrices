import pandas as pd
import yfinance as yf

df = pd.read_csv("/Users/anuarnavarro/Desktop/TFG/GitHub/ForecastStockPrices/Code/Data/NASDAQ100/NASDAQ-Symbols.csv")
symbols_string = " ".join(df['Ticker'].tolist())

data = yf.download(symbols_string, start="1990-01-01")['Close']

data.to_csv('/Users/anuarnavarro/Desktop/TFG/GitHub/ForecastStockPrices/Code/Data/NASDAQ100/NASDAQ100-Prices.csv')
