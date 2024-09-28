import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Load the dataset
my_file = pd.read_csv("C:/Users/COMTECH COMPUTER/Desktop/UDEMY_TSA_FINAL/Data/airline_passengers.csv",
                      index_col=0, parse_dates=True)

my_file.index.freq='MS'
# my_file.dropna(inplace=True)
print(my_file.head())
my_file['Thousands'].plot(label="Original Data")
# plt.show()

# Split the data
X_train = my_file['Thousands'].iloc[:96]
X_test = my_file['Thousands'].iloc[96:]
# print(X_test.head())
# X_train.plot()
# plt.show()

# Apply Holt-Winters Exponential Smoothing
hw_model = ExponentialSmoothing(X_train, trend='mul', seasonal='mul', seasonal_periods=12).fit()
# Print the fitted model
predictions = hw_model.forecast(48)
print(predictions.head())

X_train.plot()
predictions.plot(label="Fitted Data",linestyle="--")
plt.show()

"""Evaluation Matrix"""
# from sklearn.metrics import mean_squared_error,mean_absolute_error
# mean_error = mean_absolute_error(X_test,predictions)
# print(mean_error)

# mean_sq_error = mean_squared_error(X_test,predictions)
# print(mean_sq_error)
# #-> for root mean squared error
# print(np.sqrt(mean_sq_error))

"""Final model"""
x_test = my_file['Thousands']
my_model = ExponentialSmoothing(x_test,trend='mul',seasonal='mul',seasonal_periods=12).fit()
#-> future forcasting of 36 months
my_predictions = my_model.forecast(36)

x_test.plot()
my_predictions.plot()
plt.show()

# """Evaluation of Result"""
#-> not possible on future values
# from sklearn.metrics import mean_squared_error
# root_mean_sq_error = np.sqrt(mean_squared_error(x_test,my_predictions))
# print(root_mean_sq_error)


