import pandas as pd
import yfinance as yf

df = pd.read_csv("/Users/anuarnavarro/Desktop/TFG/Code/Data/S&P500/S&P500-Symbols.csv")
symbols_string = " ".join(df['Symbol'].tolist())

data = yf.download(symbols_string, start="2005-01-01")['Close']

data.to_csv('/Users/anuarnavarro/Desktop/TFG/Code/Data/S&P500/S&P500-Prices.csv')
