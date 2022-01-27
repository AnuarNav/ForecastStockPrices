import pandas as pd
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

table = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')

df = table[3]
print(df)
df.to_csv("/Users/anuarnavarro/Desktop/TFG/Code/Data/NASDAQ100/NASDAQ-Symbols.csv", columns=['Ticker'])