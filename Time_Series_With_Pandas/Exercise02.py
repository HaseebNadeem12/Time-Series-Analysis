import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

my_data02 = pd.read_csv('C:/Users/COMTECH COMPUTER/Desktop/Data/Time-series-with-pandas/UMTMVS.csv')
# print(my_data02.head())
# print(my_data02.dtypes)


"""1. Set the DATE column as the index."""
#-> Now set DATE to Index
# my_data02.set_index('DATE')
# print(my_data02.head())


"""2. Check the data type of the index."""
# print(my_data02.dtypes)

"""3. Convert the index to be a datetime index. """
# my_data02['DATE'] = pd.to_datetime(my_data02['DATE'])
# my_data02.set_index('DATE',inplace= True)
# print(my_data02.head())

"""4. Plot out the data, choose a reasonable figure size"""
# my_data02.plot(figsize= (12,6))
# plt.show()

"""5. What was the percent increase in value from Jan 2009 to Jan 2019?"""
# res =  100* (my_data02.loc['2019-01-01'] - my_data02.loc['2009-01-01']) / my_data02.loc['2009-01-01']
# print(res)

"""6. What was the percent decrease from Jan 2008 to Jan 2009?"""
# res1 = 100* (my_data02.loc['2009-01-01'] - my_data02.loc['2008-01-01'])/my_data02.loc['2008-01-01']
# print(res1)

"""7. What is the month with the least value after 2005?"""
# print(my_data02.loc['2005-12-30':].idxmin(axis=0))

"""8. What 6 months have the highest value"""
# print(my_data02.sort_values(by='UMTMVS',ascending= False).head(6))

"""9. what was the value difference between Jan 2008 and Jan 2009"""
# res2 = my_data02.loc['2008-01-01'] - my_data02.loc['2009-01-01']
# print(res2)

"""10. Create a bar plot showing the average value in millions of dollars per year"""
# res3 = my_data02.resample('Y').mean().plot.bar(figsize = (12,5))
# plt.show()


"""11. What year had the biggest increase in mean value from the previous year's mean value"""
# yearly_data = my_data02.resample('Y').mean()
# yearly_data_shift = yearly_data.shift(1)
#
# #-> Calculate the yearly average
# print(yearly_data.head())
# print(yearly_data_shift.head())
#
# change = yearly_data - yearly_data_shift
# print(change['UMTMVS'].idxmax(axis =0 ))


"""12. Plot out the yearly rolling mean on top of the original data. Recall that this is monthly data and there are 12 months in a year!"""
# my_data02.plot ( figsize = (12,6) ) # -> Actual plot
# my_data02['UMTMVS'].rolling(window=12).mean().plot()
# plt.show()

"""BONUS QUESTION (HARD)."""
# # Some month in 2008 the value peaked for that year. How many months did it take to surpass that 2008 peak?
# # (Since it crashed immediately after this peak) There are many ways to get this answer. NOTE:
# # I get 70 months as my answer, you may get 69 or 68, depending on whether or not you count the start and end months.
# # Refer to the video solutions for full explanation on this.
#
# my_file03 = pd.read_csv('C:/Users/COMTECH COMPUTER/Desktop/Data/Time-series-with-pandas/UMTMVS.csv',
#                         index_col='DATE',parse_dates= True)
# peak_value = my_file03.loc['2008-01-01':'2008-12-01']
# print(peak_value.head(7))
# print(peak_value.max())
#
# rest_data = my_file03.loc['2008-06-01':]
# res = rest_data[rest_data>=510081].dropna()
# print(res)
#
# print(len(my_file03.loc['2008-06-01':'2014-03-01']))











