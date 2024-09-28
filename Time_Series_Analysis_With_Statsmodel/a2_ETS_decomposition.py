"""Error\Trend\Seasnolity"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


my_file = pd.read_csv("C:/Users/COMTECH COMPUTER/Desktop/UDEMY_TSA_FINAL/Data/airline_passengers.csv",
                      index_col=0,parse_dates=True)
my_file.dropna(inplace=True)
print(my_file.head(15))

my_file.plot()
# my_file['Thousands of Passengers'].plot()
plt.show()

"""To seperate out Error, Trend, and Seasnolity"""
# Residual plot shows Error
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(my_file['Thousands of Passengers'], model='multiplicative' )
result.plot()

result = seasonal_decompose(my_file['Thousands of Passengers'], model='additive' )
result.plot()

plt.show()