import pandas as pd
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

table=pd.read_html('https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average')

df = table[1]

df.to_csv("/Users/anuarnavarro/Desktop/TFG/Code/Data/DJI/DJI-Symbols.csv", columns=['Symbol'])