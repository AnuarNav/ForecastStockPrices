import pandas as pd
import yfinance as yf

df = pd.read_csv("/Users/anuarnavarro/Desktop/TFG/Code/Data/DJI/DJI-Symbols.csv")
symbols_string = " ".join(df['Symbol'].tolist())

data = yf.download(symbols_string, start="2005-01-01")['Close']

data.to_csv('/Users/anuarnavarro/Desktop/TFG/Code/Data/DJI/DJI-Prices.csv')
