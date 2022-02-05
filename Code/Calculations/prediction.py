"""This module implements an LSTM neural network to predict prices of stocks in the 3 indexes (S&P, DJI,
DAX) and saves the predicted prices in 3 excels, one for each index, into
/Users/anuarnavarro/Desktop/TFG/GitHub/ForecastStockPrices/Code/Data/INDEX_NAME
/INDEX_NAME_predicted_prices.xlsx

in the following format:
| date | stock1 | stock2 | ... | """
import inline as inline
import matplotlib
import pandas as pd
import numpy as np
from Calculations import constants
import calculations
import time
# Feature Scaling for fast training of neural networks
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import tensorflow


for index in constants.indexes:
    prices_predicted_dfs = []

    for i in range(constants.years_window_size, len(constants.dates)):
        print(f'''### Predicting index {index}; Date {i - 1}/{len(constants.dates) - i} ###''')
        start_d = constants.dates[i - constants.years_window_size]
        end_d = calculations.get_one_year_later(constants.dates[i])
        prices_df = calculations.get_prices(index, False, start_d, end_d)
        assert not np.any(np.isnan(prices_df))
        for (stockTicker, prices) in prices_df.iteritems():
            prices = prices.values.reshape(-1, 1)  # List has to be 2d to be normalized
            # Choosing between Standardization or normalization
            # sc = StandardScaler()
            sc = MinMaxScaler()

            DataScaler = sc.fit(prices)
            X = DataScaler.transform(prices)  # Normalize prices (values between 0-1 to save memory)
            X = X.reshape(X.shape[0],)  # Reshape back to 1d list
            print('### After Normalization ###')
            print(X[-10:])

            # x_sample == input data AND y_sample == expected output data
            # Multi step data preparation

            # split into samples
            X_samples = list()
            y_samples = list()

            NumberOfRows = len(X)
            TimeSteps = 30  # next few day's Price Prediction is based on last how many past day's prices
            FutureTimeSteps = 250  # How many days in future you want to predict the prices (it will predict prices
            # of day 1, 2, 3, ..., 250. LSTM uses a sliding window starting from: day prediction 1, and,
            # with that prediction, then calculates prediction of day 2 (moves the startingIndex of the window by one
            # each time it does a prediction until it predicts 250).

            # Iterate through the values to create combinations
            for j in range(TimeSteps, NumberOfRows - FutureTimeSteps, 1):
                x_sample = X[j - TimeSteps:j]
                y_sample = X[j:j + FutureTimeSteps]
                X_samples.append(x_sample)
                y_samples.append(y_sample)

            ################################################
            # Reshape the Input as a 3D (samples, Time Steps, Features)
            X_data = np.array(X_samples)
            X_data = X_data.reshape(X_data.shape[0], X_data.shape[1], 1)
            print('### Input Data Shape ###')
            print(X_data.shape)

            # We do not reshape y as a 3D data as it is supposed to be a single column only
            y_data = np.array(y_samples)
            print('### Output Data Shape ###')
            print(y_data.shape)

            #  Splitting the data into Training and Testing
            # Choosing the number of testing data records
            TestingRecords = 1

            # Splitting the data into train and test
            X_train = X_data[:-TestingRecords]
            X_test = X_data[-TestingRecords:]
            y_train = y_data[:-TestingRecords]
            y_test = y_data[-TestingRecords:]

            #############################################
            # Printing the shape of training and testing
            print('\n#### Training Data shape ####')
            print(X_train.shape)
            print(y_train.shape)

            print('\n#### Testing Data shape ####')
            print(X_test.shape)
            print(y_test.shape)

            # Creating the Deep Learning Multi-Step LSTM model
            # Defining Input shapes for LSTM
            TimeSteps = X_train.shape[1]
            TotalFeatures = X_train.shape[2]
            print("Number of TimeSteps:", TimeSteps)
            print("Number of Features:", TotalFeatures)

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
            regressor.add(Dense(units=FutureTimeSteps))

            # Compiling the RNN
            regressor.compile(optimizer='adam', loss='mean_squared_error')

            ###################################################################
            # Measuring the time taken by the model to train
            StartTime = time.time()

            # Fitting the RNN to the Training set
            regressor.fit(X_train, y_train, batch_size=5, epochs=5)

            EndTime = time.time()
            print("############### Total Time Taken: ", round((EndTime - StartTime) / 60), 'Minutes #############')

            ###################
            # Getting the original price values for testing data
            orig = DataScaler.inverse_transform(y_test)
            print('\n#### Original Prices ####')
            print(orig)

            # Making predictions on the 2 first year values (first 512 values from X)
            predicted_Price = regressor.predict(X_test)
            predicted_Price = DataScaler.inverse_transform(predicted_Price)
            print('#### Predicted Prices ####')
            print(predicted_Price)

            raise ValueError("")

        prices_predicted_df = 0
        prices_predicted_dfs.append(prices_predicted_df)

    all_prices_predicted_df = pd.concat(prices_predicted_dfs).reset_index(drop=True)

    all_prices_predicted_df.to_excel(
        f'''/Users/anuarnavarro/Desktop/TFG/GitHub/ForecastStockPrices/Code/Data/{index}/{index}_predicted_prices.xlsx''')
