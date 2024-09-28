import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Load the dataset
my_file = pd.read_csv("C:/Users/COMTECH COMPUTER/Desktop/UDEMY_TSA_FINAL/Data/airline_passengers.csv",
                      index_col=0, parse_dates=True)

# my_file.index.freq='MS'
# # my_file.dropna(inplace=True)
# print(my_file.head())
# my_file['Thousands'].plot(label="Original Data")
# # plt.show()
#
# # Split the data
# X_train = my_file['Thousands'].iloc[:96]
# X_test = my_file['Thousands'].iloc[96:]
# # print(X_test.head())
# # X_train.plot()
# # plt.show()

# # Apply Holt-Winters Exponential Smoothing
# hw_model = ExponentialSmoothing(X_train, trend='mul', seasonal='mul', seasonal_periods=12).fit()
# # Print the fitted model
# predictions = hw_model.forecast(48)
# print(predictions.head())
#
# X_train.plot()
# predictions.plot(label="Fitted Data",linestyle="--")
# plt.show()

"""Evaluation Matrix"""
# from sklearn.metrics import mean_squared_error,mean_absolute_error
# mean_error = mean_absolute_error(X_test,predictions)
# print(mean_error)

# mean_sq_error = mean_squared_error(X_test,predictions)
# print(mean_sq_error)
# #-> for root mean squared error
# print(np.sqrt(mean_sq_error))

"""Final model"""
# x_test = my_file['Thousands']
# my_model = ExponentialSmoothing(x_test,trend='mul',seasonal='mul',seasonal_periods=12).fit()
# #-> future forcasting of 36 months
# my_predictions = my_model.forecast(36)
#
# x_test.plot()
# my_predictions.plot()
# plt.show()

# """Evaluation of Result"""
#-> not possible on future values
# from sklearn.metrics import mean_squared_error
# root_mean_sq_error = np.sqrt(mean_squared_error(x_test,my_predictions))
# print(root_mean_sq_error)

"""Differencing"""
"""
Stationarity: Many time series models, like ARIMA, assume that the data is stationary. Differencing helps make non-stationary data stationary.
Trend Removal: Differencing removes trends in the data, making it easier to model the short-term dynamics of the time series.
Improves Model Accuracy: By eliminating trends, differencing can improve the accuracy of your time series forecasting models.
"""
#-> used to convert non-stationary data into stationary data
my_file01 = pd.read_csv("C:/Users/COMTECH COMPUTER/Desktop/UDEMY_TSA_FINAL/Data/samples.csv")
print(my_file01.head())

#-> It's a stationary data
my_file01['a'].plot()
plt.show()

#-> It's a non-stationary data
my_file01['b'].plot()
plt.show()

#-> Converting non-stationary data into stationary data by differencing
first_order_diff = my_file01['b'] - my_file01['b'].shift(1)
first_order_diff.plot()
plt.show()

#-> another way
from statsmodels.tsa.statespace.tools import diff
first_order_difference = diff(my_file01['b'], k_diff=1)
second_order_difference = diff(my_file01['b'], k_diff=2)

print(first_order_difference)
print(second_order_difference)



