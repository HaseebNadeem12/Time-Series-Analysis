"""Holt Winter's Method"""
"""For second and third fector calculation"""

"""
In this section we'll look at Double and Triple Exponential Smoothing with the Holt-Winters Methods.

n Double Exponential Smoothing (aka Holt's Method) we introduce a new smoothing factor 
 (beta) that addresses trend:
 
 With Triple Exponential Smoothing (aka the Holt-Winters Method) we introduce a smoothing factor 
 (gamma) that addresses seasonality:

Here 
 represents the number of divisions per cycle. In our case looking at monthly data that displays a repeating pattern each year, we would use 
.

In general, higher values for 
, 
 and 
 (values closer to 1), place more emphasis on recent data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# my_file_02 = pd.read_csv("C:/Users/COMTECH COMPUTER/Desktop/UDEMY_TSA_FINAL/Data/airline_passengers.csv")
my_file_02 = pd.read_csv("C:/Users/COMTECH COMPUTER/Desktop/Book1.csv",
                           index_col=0, parse_dates=True)
# print(my_file_02.index)
# print(my_file_02.head())
#  Setting frequency as Monthly Start
#  In order to build a Holt-Winters smoothing model, statsmodels needs
#  to know the frequency of the data (whether it's daily, monthly etc.).
my_file_02.index.freq = 'MS'
# print(my_file_02.index)
# print(my_file_02.head())

my_file_02['Consumption'] = my_file_02['Consumption'].str.replace(',', '').astype(float)

"""Simple Exponential Smoothing"""
"""
A variation of the statmodels Holt-Winters function provides Simple Exponential Smoothing. We'll show that it performs the same calculation of the weighted moving average as the pandas .ewm() method:
"""
span = 12
alpha = 2/(span+1)
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
# Both are same
my_file_02['EWMA'] = my_file_02['Consumption'].ewm(span = 12, adjust=False).mean()
my_file_02['SES'] = SimpleExpSmoothing(my_file_02['Consumption']).fit(smoothing_level=alpha, optimized = False).fittedvalues.shift(-1)

# print(my_file_02.head())
# my_file_02.plot()
# plt.show()

"""Double Exponential Smoothing"""
"""
Where Simple Exponential Smoothing employs just one smoothing factor 
 (alpha), Double Exponential Smoothing adds a second smoothing factor 
 (beta) that addresses trends in the data. Like the alpha factor, values for the beta factor fall between zero and one (
). The benefit here is that the model can anticipate future increases or decreases where the level model would only work from recent calculations.
"""
from statsmodels.tsa.holtwinters import ExponentialSmoothing

#-> With Additive property(linear)
my_file_02['DoubleES_add'] = ExponentialSmoothing(my_file_02['Consumption'],trend='add').fit().fittedvalues.shift(-1)

# my_file_02["1959-05-01":].plot()
my_file_02[['Consumption','DoubleES_add']].plot(figsize=(12,8))
# my_file_02[['Thousands of Passengers','EWMA','DoubleES_add']].iloc[:24].plot(figsize=(12,8))
# plt.show()

#-> With Multiplicative property(non_linear)
my_file_02['DoubleES_mul'] = ExponentialSmoothing(my_file_02['Consumption'], trend= 'mul').fit().fittedvalues.shift(-1)
my_file_02[['Consumption','DoubleES_mul']].plot(figsize=(12,8))
# plt.show()


"""Tripple Exponential Smoothing"""
"""
Triple Exponential Smoothing, the method most closely associated with Holt-Winters, adds support for both trends and seasonality in the data.
"""

from statsmodels.tsa.holtwinters import ExponentialSmoothing
my_file_02['TrippleES'] = ExponentialSmoothing(my_file_02['Consumption'],trend='add',seasonal='add', seasonal_periods=12).fit().fittedvalues
print(my_file_02.head())
my_file_02[['Consumption','TrippleES']].plot(figsize = (12,8))
plt.show()


