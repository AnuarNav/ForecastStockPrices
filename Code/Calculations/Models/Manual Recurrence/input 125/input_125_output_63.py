"""This module implements an LSTM neural network to predict prices of stocks in the 3 indexes (S&P, DJI,
DAX) and saves the predicted prices in 3 excels, one for each index, into
/Users/anuarnavarro/Desktop/TFG/GitHub/ForecastStockPrices/Code/Data/INDEX_NAME
/INDEX_NAME_predicted_prices.xlsx

in the following format:
| date | stock1 | stock2 | ... |

Predictions only done until year 2019 (included) as, even though 2020 data is available, """
import os.path
import pandas as pd
import numpy as np
from Calculations import constants, calculations
import time
# Feature Scaling for fast training of neural networks
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

input_ = 'input 125'
months = constants.trimester_months
dates = constants.trimester_dates
window_size = constants.trimester_window_size
recurrence = 'Manual Recurrence'
TimeSteps = 125  # next few day's Price Prediction is based on last how many past day's prices -INPUT
FutureTimeSteps = 63  # How many days in future you want to predict the prices (it will predict prices
# of day 1, 2, 3, ..., 250. LSTM uses a sliding window starting from: day prediction 1, and,
# with that prediction, then calculates prediction of day 2 (moves the startingIndex of the window by one
# each time it does a prediction until it predicts 250). - OUTPUT

TotalStartTime = time.time()

for index in constants.indexes:
    # Get all prices from [2005-2020] (both included)
    prices_df = calculations.get_prices(index, False, dates[0], calculations.get_x_months_later_date(
        dates[len(dates) - 1], months)).dropna()

    # df where all stocks of an index will be stored
    all_index_stock_prices_predicted_df = pd.DataFrame()

    stock_index = 0
    for (stockTicker, all_prices) in prices_df.iteritems():
        stock_index += 1
        list_single_stock_prices_predicted_dfs = []  # List that contains all prices_predicted_df's
        prices_predicted_df = pd.DataFrame()  # Contains prices predicted between specific dates of a single stock
        for i in range(window_size, len(dates)):
            prices_predicted_df = pd.DataFrame()

            print(f'''### Predicting index {index}; Stock {stockTicker} ({stock_index}/{len(prices_df.columns)}); Date {i - window_size + 1}/{len(dates) - window_size} ###''')

            # Get only prices between start and end dates into a dataframe
            start_d = dates[i - window_size]
            end_d = dates[i]
            stock_prices_bet_dates_df = prices_df.loc[start_d: end_d]  # Stocks b4 prices to predict
            end_d_predict = calculations.get_x_months_later_date(end_d, months)
            stock_prices_bet_dates_and_to_predict_df = prices_df.loc[start_d: end_d_predict]
            # Get prices to predict (to get dates to predict easily afterwards
            start_d_predict = end_d
            stock_prices_to_predict_df = prices_df.loc[start_d_predict: end_d_predict]

            # Get specific stock column and convert it to 2D list, so it can be normalized
            stock_prices_bet_dates_and_to_predict = stock_prices_bet_dates_and_to_predict_df[stockTicker].values.reshape(-1, 1)
            stock_prices_bet_dates = stock_prices_bet_dates_df[stockTicker].values.reshape(-1, 1)

            # Choosing between Standardization or normalization
            # sc = StandardScaler()
            sc = MinMaxScaler()

            # Fit all prices but only train with prices b4 prices to predict
            DataScaler = sc.fit(stock_prices_bet_dates_and_to_predict)
            X = DataScaler.transform(stock_prices_bet_dates)  # Normalize prices (values between 0-1 to save memory)
            X = X.reshape(X.shape[0], )  # Reshape back to 1d list

            # split into samples
            X_samples = list()
            y_samples = list()

            NumberOfRows = len(X)

            # Iterate through the values to create combinations
            for j in range(TimeSteps, NumberOfRows - constants.manual_future_time_steps, 1):
                x_sample = X[j - TimeSteps:j]
                y_sample = X[j:j + constants.manual_future_time_steps]
                X_samples.append(x_sample)
                y_samples.append(y_sample)

            ################################################
            # Reshape the Input as a 3D (samples, Time Steps, Features)
            X_data = np.array(X_samples)
            X_data = X_data.reshape(X_data.shape[0], X_data.shape[1], 1)
            # print('### Input Data Shape ###')
            # print(X_data.shape)

            # We do not reshape y as a 3D data as it is supposed to be a single column only
            y_data = np.array(y_samples)
            # print('### Output Data Shape ###')
            # print(y_data.shape)

            #############################################
            # Printing the shape of training and testing
            # print('\n#### Training Data shape ####')
            # print(X_data.shape)
            # print(y_data.shape)

            # print('\n#### Testing Data shape ####')
            # print(X_test.shape)
            # print(y_test.shape)

            # Creating the Deep Learning Multi-Step LSTM model
            # Defining Input shapes for LSTM
            TotalFeatures = X_data.shape[2]
            # print("Number of TimeSteps:", TimeSteps)
            # print("Number of Features:", TotalFeatures)

            # Initialising the RNN
            regressor = Sequential()

            # Adding the First input hidden layer and the LSTM layer
            # return_sequences = True, means the output of every time step to be shared with hidden next layer
            regressor.add(
                LSTM(units=10, activation='relu', input_shape=(TimeSteps, TotalFeatures), return_sequences=True))

            # Adding the Second hidden layer and the LSTM layer
            regressor.add(
                LSTM(units=5, activation='relu', input_shape=(TimeSteps, TotalFeatures), return_sequences=True))

            # Adding the Third hidden layer and the LSTM layer
            regressor.add(LSTM(units=5, activation='relu', return_sequences=False))

            # Adding the output layer
            # Notice the number of neurons in the dense layer is now the number of future time steps
            # Based on the number of future days we want to predict
            regressor.add(Dense(units=constants.manual_future_time_steps))

            # Compiling the RNN
            regressor.compile(optimizer='adam', loss='mean_squared_error')

            ###################################################################
            # Measuring the time taken by the model to train
            StartTime = time.time()

            # Fitting the RNN to the Training set
            regressor.fit(X_data, y_data, batch_size=constants.batch_size, epochs=constants.epochs)

            EndTime = time.time()
            # print("############### Total Time Taken: ", round((EndTime - StartTime) / 60), 'Minutes #############')

            ###################
            prices_predicted = []  # List to store prices predicted (ultimate size will be equal to FutureTimeSteps)
            lastXDaysPrices = DataScaler.transform(stock_prices_bet_dates[-TimeSteps:])  # Store last X days prices
            # from Y next prices to be predicted
            curr_total_future_time_steps = 0

            while curr_total_future_time_steps < FutureTimeSteps:
                lastXDaysPrices_3d = lastXDaysPrices.reshape(1, TimeSteps, 1)
                x_next_predicted_Price = regressor.predict(lastXDaysPrices_3d)

                # Append transformed predicted to prices to previous prices to predict next manual_future_time_steps
                lastXDaysPrices = np.append(lastXDaysPrices, x_next_predicted_Price[0])
                # Get only previous manual_future_time_steps
                lastXDaysPrices = lastXDaysPrices[constants.manual_future_time_steps:]

                # Append untransformed predicted prices to list with all predicted
                x_next_predicted_Price = DataScaler.inverse_transform(x_next_predicted_Price)[0]
                prices_predicted.extend(x_next_predicted_Price.tolist())
                # Add manual_future_time_steps to curr
                curr_total_future_time_steps += constants.manual_future_time_steps

            # Crete df from dates and predicted prices
            dates_predicted = stock_prices_to_predict_df.index.values
            prices_predicted = prices_predicted[:len(dates_predicted)]  # Get rid of over-predicted values
            prices_predicted_df = pd.DataFrame({stockTicker: prices_predicted, 'Date': dates_predicted})
            prices_predicted_df.set_index('Date', inplace=True)
            # print(prices_predicted_df)  # HERE
            list_single_stock_prices_predicted_dfs.append(prices_predicted_df)

        # Add all dates into a single df
        single_stock_all_prices_predicted_df = pd.concat(list_single_stock_prices_predicted_dfs)\
            .groupby(level=0).last()  # Drop duplicated indexes as last date of each interval overlap
        # Add all dates of a single stock into the index df which includes all
        all_index_stock_prices_predicted_df = all_index_stock_prices_predicted_df.join(
            single_stock_all_prices_predicted_df) if not all_index_stock_prices_predicted_df.empty else \
            single_stock_all_prices_predicted_df

    print(f'''Index {index} predictions df:''')
    print(all_index_stock_prices_predicted_df)
    # Save df which contains all predictions from [2007-2020] of all stocks in current index
    all_index_stock_prices_predicted_df.groupby(level=0).last().to_excel(
        f'''/Users/anuarnavarro/Desktop/TFG/GitHub/ForecastStockPrices/Code/Data/{index}/PredictedPrices/{recurrence}/{input_}/{index}_predicted_prices.xlsx''')

TotalEndTime = time.time()
total_time_taken_df = pd.DataFrame({'TimeTaken in Minutes': round((TotalEndTime - TotalStartTime) / 60), 'TimeTaken in Hours': round((TotalEndTime - TotalStartTime) / 360)}, index=[0])
total_time_taken_df.to_excel(f'''/Users/anuarnavarro/Desktop/TFG/GitHub/ForecastStockPrices/Code/Data/TimeTaken/{recurrence}/{input_}/{os.path.basename(__file__)}_time.xlsx''')

