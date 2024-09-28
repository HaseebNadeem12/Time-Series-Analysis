import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#->Creating date colomn to datetimeindex
my_data = pd.read_csv('C:/Users/COMTECH COMPUTER/Desktop/Data/Time-series-with-pandas/starbucks.csv',
                      index_col='Date',parse_dates=True)

"""resample()"""
# #->A common operation with time series data is resampling based on the time series index.
# print(my_data.head(),type(my_data))
# print(my_data.resample(rule='YE').mean())  # just Calculate mean of 12 months And give 1 value of that year

"""Custome Resempling"""
# def first_day(entry):
#     """
#     Returns the first instance of the period, regardless of sampling rate.
#     """
#     if len(entry):  # handles the case of missing data
#         return entry[0]
#
# # my_data['Close'].resample(rule='YE').apply(first_day).plot.line(title = 'My Data',figsize = (12,5))
# print(my_data['Close'].resample('ME').max.head())
# print(my_data['Close'].head())

"""Rolling & Expanding"""
# # Rolling
# print(my_data['Close'].plot(figsize=(12,5)).autoscale(axis='x',tight=True))
# print(my_data['Close'].rolling(window=7).mean().plot())
# print(my_data['Close'].rolling(window=30).mean().plot())

# Expamding
#-> It include all the previous points as well as the current point, thats why it is called as expanding
# print(my_data['Close'].expanding(min_periods= 30).mean().plot())

plt.show()

