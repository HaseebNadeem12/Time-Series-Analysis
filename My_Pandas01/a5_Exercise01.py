import pandas as pd
import numpy as np

"""1. Import pandas and read in the population_by_county.csv file into a dataframe called pop."""
# Full path to the Excel file on your desktop
file_path = r'C:\Users\COMTECH COMPUTER\Desktop\population_by_county.csv'

# Read the Excel file
pop = pd.read_csv(file_path)

"""2. Show the head of the dataframe"""
print(pop.head())

"""3. What are the column names?"""
# print(pop.columns)

"""4. How many unique States are represented in this data set?"""
# print(pop['State'].nunique())  # Return the number of Unique State

"""5. Get a Unique list or array of all the states in the data set."""
# print(pop['State'].unique())    # Return the list of Unique State

"""6. What are the five most common County names in the U.S.?"""
print(pop['County'].value_counts())         #Return Most Number of cities in ascending order
print(pop['County'].value_counts().head())  #Return first 5 Most Number of cities in ascending order

"""7. What are the top 5 most populated Counties according to the 2010 Census?"""
# # print(pop.value_counts().head())    # not the correct one(Sort first in ascending order)
# print(pop.sort_values('2010Census',ascending=False).head())

"""8. What are the top 5 most populated States according to the 2010 Census?"""
# print(pop.groupby('State').sum().sort_values('2010Census',ascending=False).head())  #not correct

"""9. How many Counties have 2010 populations greater than 1 million?"""
# print(len(pop[pop['2010Census'] > 1000000 ]))

"""10. How many Counties don't have the word "County" in their name?"""
# # one way to do the question
# def Check_County(name):
#     res = 'County' not in name
#     return res
# ans = pop['County'].apply(Check_County)
# print(sum(ans))

# Using labdas function
# print(sum(pop['County'].apply(lambda name: 'County' not in name)))


"""11. Add a column that calculates the percent change between the 2010 Census and the 2017 Population Estimate"""
# pop['Percentage_change'] = 100 *(pop['2017PopEstimate'] - pop['2010Census'])/pop['2010Census']
# print(pop.head())


"""Bonus: What States have the highest estimated percent change between the 2010 Census and the 2017 Population Estimate?
This will take several lines of code, as it requires a recalculation of PercentChange."""

# STEP 1: GROUP BY STATE, TAKE THE SUM OF POP COUNTY
# States = pop.groupby('State').sum()

# STEP 2: RECALCULATE THE PERCENTAGE CHANGES (based off the total sum change per state)
# States['PercentChange'] = 100 * (States['2017PopEstimate'] - States['2010Census']) / States['2010Census']

# STEP 3: SORT VALUES BY NEW PERCENT CHANGE
# print(States.sort_values('PercentChange', ascending=False).head())