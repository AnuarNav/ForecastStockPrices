"""Gets the efficient portfolios for each {quarter} based on each 2 year window previous to {quarter} + next {
quarter} of predicted prices (results of  prediction saved in
/Users/anuarnavarro/Desktop/TFG/GitHub/ForecastStockPrices/Code/Data/INDEX_NAME/PredictedPrices/{RECURRENCE}/{INPUT}/
INDEX_NAME_predicted_prices_output_{NUMBER_OF_PRICES}.xlsx.) using Markowitz and Sharpe ratio based on previous prices
for all indexes (DJI,DAX, S&P) AND calculates the return of each portfolio. Portfolio for next QUARTER.

For each index AND each INPUT AND AUTO/MANUAL Recurrence size in prediction it saves a new excel file into:
/Users/anuarnavarro/Desktop/TFG/GitHub/ForecastStockPrices/Code/Data/INDEX_NAME/Efficient Portfolios with
Prediction/{INPUT}/INDEX_NAME_efficient_portfolios_and_returns_with_prediction_{quarter}.xlsx

Result format (2 rows for each window, Markowitz and Sharpe):
| Strategy (Markowitz||Sharpe) | start_date | end_date | Stock1 weight | Stock2 weight | ...  | Return
"""