import pandas as pd
import numpy as np
from datetime import datetime

"""Python Datetime Review"""
# # Creating some variable
# my_year = 2001
# my_month = 7
# my_day = 12
# my_hour = 13
# my_minute = 23
# my_second = 43
#
# my_date = datetime(my_year,my_month,my_day)
# print(my_date)
# print(my_date.day)
# my_date_time = datetime(my_year,my_month,my_day,my_hour,my_minute,my_second)
# print(my_date_time)
# print(my_date_time.second)

"""NumPy Datetime Arrays"""
# # -> We mentioned that NumPy handles dates more efficiently than Python's datetime format.
# # -> The NumPy data type is called datetime64 to distinguish it from Python's datetime.
#
# my_array1 = np.array(['2001-07-12','2000-10-19','2001-09-18'], dtype = 'datetime64[D]')
# my_array2 = np.array(['2001-07-12','2000-10-19','2001-09-18'], dtype = 'datetime64[Y]')
# my_array3 = np.array(['2001-07-12','2000-10-19','2001-09-18'], dtype = 'datetime64[h]')
# # 'datetime64[D]'. This tells us that NumPy applied a day-level date precision.
# print(my_array1,type(my_array1))
# print(my_array2,type(my_array2))
# print(my_array3,type(my_array3))

"""NumPy Date Ranges"""
# -> we got an array of date
my_range = np.arange('2001-07-12','2001-08-12','2',dtype='datetime64[D]')
print(my_range,type(my_range))

"""Pandas Datetime Index"""
#->The simplest way to build a DatetimeIndex is with the pd.date_range() method:
my_date_range = pd.date_range('7/8/2018', periods=20, freq='D')
result = pd.DatetimeIndex(my_date_range)
print(result,type(result))

# ->Another way is to pass a list or an array of datetime objects into the pd.DatetimeIndex() method:
my_date_range1 = np.array(['2001-08-12','2008-09-19','2015-10-29'],dtype='datetime64')
result1 = pd.DatetimeIndex(my_date_range1)
print(result1,type(result1))

"""Pandas Datetime Analysis"""
data1 = np.random.randn(3,2)
col = ['A','B']
result1 = pd.DataFrame(data1,my_date_range1,col)
print(result1)

print(result1.index)
print(result1.index.max())
print(result1.index.min())
print(result1.index.argmax())