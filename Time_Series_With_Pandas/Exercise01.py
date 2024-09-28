import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.conftest import index

# my_file = pd.read_csv('C:/Users/COMTECH COMPUTER/Desktop/Data/Time-series-with-pandas/monthly_milk_production.csv')
# print(my_file.head())

"""1. What is the current data type of the Date column?"""
# print(my_file.dtypes)

"""2. Change the Date column to a datetime format"""
# my_file['Date'] = pd.to_datetime(my_file['Date'])
# print(my_file.head(2))
# print(my_file.columns)

"""3. Set the Date column to be the new index"""
# my_file.set_index('Date', inplace=True)
# print(my_file.head(2))
# print(my_file.columns)

"""4. Plot the DataFrame with a simple line plot. What do you notice about the plot?"""
# my_file['Production'].plot(figsize= (12,5))
# plt.show()

"""5. Add a column called 'Month' that takes the month value from the index"""
# print(my_file.index.month)
# #-> Date colomn must be in datetimeformat
# my_file['Month'] = my_file.index.month
# print(my_file.head(3))
#
#   BONOUS QUESTION
#-> Changing month in number format to name format
# my_file['Month']= my_file.index.strftime('%B')
# print(my_file.head())

"""6. Create a BoxPlot that groups by the Month field"""
# my_file.boxplot(by= 'Month', figsize= (12,5))
# my_file['Month'].plot(figsize= (12,5))

# plt.show()




