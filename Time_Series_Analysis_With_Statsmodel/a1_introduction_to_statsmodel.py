import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

my_fil = pd.read_csv("C:/Users/COMTECH COMPUTER/Desktop/UDEMY_TSA_FINAL/Data/macrodata.csv",index_col=0, parse_dates=True)
print(my_fil.head())

"""plotting"""
# my_fil['realgdp'].plot(figsize = (12,8))
# plt.show()

"""Using Statsmodel to get the trend"""

"""
Hod rick-Prescott filter

The Hodrick-Prescott filter separates a time-series 
 into a trend component 
 and a cyclical component 
 """

from statsmodels.tsa.filters.hp_filter import hpfilter
gdp_cyclic, gdp_trend = hpfilter(my_fil['realgdp'], lamb=1600)
print(gdp_cyclic, gdp_trend)

my_fil['trend'] = gdp_trend
my_fil[['trend','realgdp']].plot(figsize=(12,8))
plt.show()

# my_fil['cycle'] = gdp_cyclic
# my_fil[['cycle','realgdp']].plot(figsize=(12,8))
# plt.show()

# for perticular time period
my_fil[['trend','realgdp']]['2008-01-31':].plot()
plt.show()


