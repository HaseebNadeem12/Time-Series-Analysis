"""Exponentially Weighted Moving Averages"""
"""Comparision between Moving Average & Exponentially Weighted Moving Averages"""

"""
pandas.DataFrame.rolling(window)   Provides rolling window calculations
pandas.DataFrame.ewm(span)         Provides exponential weighted functions
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

my_file_01 = pd.read_csv("C:/Users/COMTECH COMPUTER/Desktop/UDEMY_TSA_FINAL/Data/airline_passengers.csv",
                      index_col=0,parse_dates=True)
my_file_01.dropna(inplace=True)
print(my_file_01.head())
my_file_01.plot()
plt.show()

"""SMA, Simple Moving Average"""
# # This method has larger error
# # We are not be able to catch the trends properly
# my_file_01['6_months'] = my_file_01['Thousands of Passengers'].rolling(window=6).mean()
my_file_01['12_months'] = my_file_01['Thousands of Passengers'].rolling(window=12).mean()
# my_file_01[['Thousands of Passengers','6_months','12_months']].plot()
print(my_file_01.head())
my_file_01.plot(figsize=(12,8))
plt.show()

"""EWMA, Exponentionally weighted moving average"""
# EWMA will allow us to reduce the lag effect from SMA and
# it will put more weight on values that occured more
# recently (by applying more weight to the more recent
# values, thus the name).

# Applying Simple exponentionally smoothing with fector alpha
my_file_01['EWMA'] = my_file_01['Thousands of Passengers'].ewm(span=12, adjust=False).mean()
my_file_01[['EWMA','Thousands of Passengers']].plot()
plt.show()

#comparision between MA and simple Exponentionally Smoothing of 12 month span
my_file_01[['Thousands of Passengers','EWMA','12_months']].plot()
plt.show()


