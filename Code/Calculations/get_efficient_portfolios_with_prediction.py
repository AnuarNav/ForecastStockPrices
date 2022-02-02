"""Gets the efficient portfolios for each 2 year window from [2005...2020) + 1 year using predicted prices (results
of  prediction saved in /Users/anuarnavarro/Desktop/TFG/GitHub/ForecastStockPrices/Code/Data/INDEX_NAME/
INDEX_NAME_predicted_prices.xlsx.) using Markowitz and Sharpe ratio based on previous prices for all indexes (DJI,
DAX, S&P) AND calculates the return of each portfolio

For each index it saves a new excel file into
/Users/anuarnavarro/Desktop/TFG/GitHub/ForecastStockPrices/Code/Data/INDEX_NAME
/INDEX_NAME_efficient_portfolios_and_returns_with_prediction.xlsx

Result format (2 rows for each window, Markowitz and Sharpe):
| Strategy (Markowitz||Sharpe) | start_date | end_date | Stock1 weight | Stock2 weight | ...  | Return
"""