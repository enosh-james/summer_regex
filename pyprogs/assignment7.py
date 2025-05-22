'''

# 1.) Read a csv file in pandas and select 1 column and count how many time the value is present here

# 2.) Count how many number of unique value you have

# 3.) Drop the duplicates values from the column

# 4.) Change the integer column to float sum float datatype

# 5.) Count how many movie and tv shows are present in dataset

# 6.) Count out how many movies are released in each year

# 7.) Count out how many action movies are released between september and october 

# 8.) Find out how many numbwer of movies and tv shows were released each year


Ans 1.)

import pandas as pd

df = pd.read_csv('your_file.csv')  

column_values = df['ColumnName']  

value_counts = column_values.value_counts()

print(value_counts)
