import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load a non-stationary dataset
df1 = pd.read_csv('C:/Users/COMTECH COMPUTER/Desktop/UDEMY_TSA_FINAL/Data/airline_passengers.csv', index_col='Month', parse_dates=True)
df1.index.freq = 'MS'
print(df1.head())

# Load a stationary dataset
df2 = pd.read_csv('C:/Users/COMTECH COMPUTER/Desktop/UDEMY_TSA_FINAL/Data/DailyTotalFemaleBirths.csv', index_col='Date', parse_dates=True)
df2.index.freq = 'D'
print(df2.head())

"""pmdarima Auto-ARIMA"""
#-> Upgrade the numpy library (2.26)
from pmdarima import auto_arima

# Fit the auto_arima model
# Display the summary of the model
arima_model = auto_arima(df2['Births'])
print(arima_model.summary())
# help(auto_arima)

#-> It will also give us the order p,d, and q but here we include some more parameter
# stepwise_fit = auto_arima(df2['Births'], start_p=0, start_q=0,
#                           max_p=6, max_q=3, m=12,
#                           seasonal=False,
#                           d=None, trace=True,
#                           error_action='ignore',   # we don't want to know if an order does not work
#                           suppress_warnings=True,  # we don't want convergence warnings
#                           stepwise=True)           # set to stepwise


# stepwise_fit01 = auto_arima(df1['Thousands'], start_p=1, start_q=1,
#                           max_p=3, max_q=3, m=12,
#                           start_P=0, seasonal=True,
#                           d=None, D=1, trace=True,
#                           error_action='ignore',   # we don't want to know if an order does not work
#                           suppress_warnings=True,  # we don't want convergence warnings
#                           stepwise=True)           # set to stepwise
from statsmodels.tsa.stattools import arma_order_select_ic
# help(arma_order_select_ic)
print(arma_order_select_ic(df2['Births']))
print(arma_order_select_ic(df2['Thousands']))




