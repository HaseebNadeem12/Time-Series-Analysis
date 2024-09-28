"""Missing Values with Python"""
import pandas as pd
import numpy as np

# # creating dictionary
# Data_Frame = {'A':[1,2,np.nan],'B':[1,np.nan,np.nan],'C':[1,2,3]}
#
# My_Data_Frame = pd.DataFrame(Data_Frame)
# print(My_Data_Frame)
# print('-'*50)
#
# """Removing MIssing value"""
# print(My_Data_Frame.dropna())
# print(My_Data_Frame.dropna(axis=1))           #wamt to drop colounm
# print(My_Data_Frame.dropna(thresh=2))         #drop thoes rowwhich have 2 missing values
#
# print(My_Data_Frame.fillna('Mis'))
# print('-'*50)
#
# print(My_Data_Frame['A'].fillna(My_Data_Frame['A'].mean()) )   #filling missing values of single row