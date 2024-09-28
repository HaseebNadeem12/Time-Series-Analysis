import pandas as pd

# # creating dictionary
# data = {'company':['Google','Google','Facebook','Facebook','MSFT','MSFT'],
#         'person':['sara','ali','haseeb','adnan','yasir','laiba'],
#         'sales':[200,120,340,124,243,350]}

# # Dictionary
# print(data)
# # Converting Dictionary into DataFrame
# result = pd.DataFrame(data)
# print(result)
# print('-'*50)

# result1 = result.groupby('company').max()   # many operation can apply
# print(result1)
# print('-'*50)
#
# result2 = result.groupby('company').describe()  #gives bundle of information
# print('\n',result2)
# print('-'*50)
#
# result3 = result.groupby('company').describe().transpose()  #gives information in batter way
# print('\n',result3)
