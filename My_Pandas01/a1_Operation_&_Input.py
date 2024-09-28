"""Common Operations with Pandas"""

import pandas as pd
import numpy as np

Data_Frame = {'col1':[1,2,3,4],'col2':[444,555,666,444],'col3':['abc','def','ghi','xyz']}
Out_put = pd.DataFrame(Data_Frame)
print(Out_put)
print('-'*50)

# print(Out_put.columns)
# print(Out_put.index)
# print('-'*50)

# Unique_value = Out_put['col2'].unique()   # return array of unique values
# print(Unique_value)
# print('-'*50)

# Unique_value_1 = Out_put['col2'].nunique()  # return number of unique values
# print(Unique_value_1)
# print('-'*50)

# print(Out_put['col2'].value_counts())     # Also tell how many times a value is appearing
# print('-'*50)

# conditional statement
# print(Out_put[ (Out_put['col1'] > 2) & (Out_put['col2'] > 444) ])
# print('-'*100)

# Applying function to colounm
# def My_fun(Number):
#     res = Number ** 2
#     return res
#
# # print(Out_put['col1'].apply(My_fun))
# # print('-'*100)
#
# Out_put['New'] = Out_put['col1'].apply(My_fun)     # creating new colounm
# print(Out_put)
# print('-'*100)
#
# print( Out_put['New'].dropna() )                      # Drop missing values
# Out_put.drop('New', axis=1 , inplace=True)      # Delete column permanently
# print( Out_put )                                      # Will delete the column permanently
# print("-"*50)

"""Sorting colounm in matrix"""
# print(Out_put.sort_values('col2'))                      #Ascending order
#
# print(Out_put.sort_values('col2',ascending=False))  #Decending order
#
# # Read the Excel file
# df = pd.read_excel('C:/Users/COMTECH COMPUTER/Desktop/Factory_data01.xlsx')
# print(df.head())
# # Converting to Dataframe
# ds = pd.DataFrame(df)
# print(ds.head())

# # Read HTML file
# df1 = pd.read_html('http://www.fdic.gov/bank/individual/failed/banklist.html')
# print(df1[0].head())
