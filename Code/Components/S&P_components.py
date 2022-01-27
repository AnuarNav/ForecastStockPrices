import pandas as pd
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

table=pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')

df = table[0]
df.to_csv("/Users/anuarnavarro/Desktop/TFG/Code/Data/S&P500/S&P500-Symbols.csv", columns=['Symbol'])
