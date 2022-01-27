import pandas as pd
import yfinance as yf

df = pd.read_csv("/Users/anuarnavarro/Desktop/TFG/GitHub/ForecastStockPrices/Code/Data/DAX30/DAX30-Symbols.csv")
symbols_string = " ".join(df['Ticker symbol'].tolist())

data = yf.download(symbols_string, start="2005-01-01")['Close']

data.to_csv('/Users/anuarnavarro/Desktop/TFG/GitHub/ForecastStockPrices/Code/Data/DAX30/DAX30-Prices.csv')
