import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import xlabel

my_data01 = pd.read_csv('C:/Users/COMTECH COMPUTER/Desktop/Data/Time-series-with-pandas/starbucks.csv'
                        ,index_col='Date',parse_dates=True)

# print(my_data01.head())
# print(my_data01['Volume'].plot())
# print(my_data01['Close'].plot())

"""Adding X, Y lables"""
# ans = my_data01['Close'].plot(title= 'My Data')
# ans.set(xlabel= 'X AXIS', ylabel= 'Y AXIS')

"""X axis limit by Slicing"""
# my_data01['Close']['2017-07-12':'2017-10-12'].plot(figsize= (12,5), ls= '--',c= 'red')

"""Changing X axis format and appearance"""
# #->import matplot lib (dates)
# from matplotlib import dates
# res = my_data01['Close'].plot(xlim=['2017-07-12','2017-10-12'] )
# #-> REMOVE PANDAS DEFAULT "Date" LABEL
# res.set(xlabel= '')
# #-> SET THE TICK LOCATOR AND FORMATTER FOR THE MAJOR AXIS
# #NOTE: we passed a rotation argument rot=0 into df.plot() so that the major axis values appear horizontal, not slanted.
# res.xaxis.set_major_locator((dates.WeekdayLocator(byweekday=0) ))
# res.xaxis.set_major_formatter(dates.DateFormatter("%a-%B-%d"))


"""Major vs. Minor Axis Values """
# from matplotlib import dates
# res = my_data01['Close'].plot(xlim=['2017-07-12','2017-10-12'] )
# res.set(xlabel= '')
#
# res.xaxis.set_major_locator(dates.WeekdayLocator(byweekday=0))
# res.xaxis.set_major_formatter(dates.DateFormatter('%d'))
#
# res.xaxis.set_minor_locator(dates.MonthLocator())
# res.xaxis.set_minor_formatter(dates.DateFormatter('\n\n%b'))

"""Adding Grid line"""
# res.yaxis.grid(True)
# res.xaxis.grid(True)

plt.show()

