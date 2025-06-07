'''

#LCM of 4 and 6
a=2
b=4
if(a>b):
 greater=a
else:
 greater=b

print(greater)

while(True):
 if(greater%a==0 and greater%b==0):
  print('greater is the lcm : ',greater)
  break
 else:
  greater+=1



# String Palindrome
# String => starting,last same

#To access first and last character then we use indexing, we compare indexes
#eg. 'sarts' then 0 and 4th index should be same.




s='saras'
start=0
end=len(s)-1
a=0

while(start<end):
 if(s[start] != s[end]):
  a=1                                                   #indicator variable : tells when a condition is true or false
  break
  
 start+=1
 end-=1

if(a==1):
 print('String is not a Palindrome')
else:
 print('Palindrome')
  




# LIST => collection of element
# where element are stored on index position
# mutable datatype
# change update, delete can be performed

mylist=[1,20,30,'abc']
print(mylist[2])

mylist[0]='tiger' #update
mylist.append('enosh')  #insert at last

print(mylist)
mylist.insert(1,'Joker')  #insert at position

print(mylist)

mylist.pop()  #deletes from the last
print(mylist)




mylist=[1,20,30,51,43]
for i in range(0,len(mylist)):
 if(mylist[i]%2==0):
  print('Even : ',mylist[i])




mylist=[1,20,30,51,43]
max=mylist[0]
for i in mylist:
 if(i>max):
  max=i

print('max')





for i in range(1,4):
 print('Student',i)
 for j in range(1,i+1):
  print('Good morning',j)


-----------------------------------------------------------------------------------



14.05.2025

mylist=[50,1,10,40,2,4,12,8,24,7]
for i in range(0,len(mylist)):
 for j in range(i+1,len(mylist)):
  if(mylist[i]+mylist[j]==9):
   print('Sum of two value : ',mylist[i],mylist[j])


mytuple=(10,20,30,'abc',10,50)
mytuple.count(10)  #count the occurence
mytuple.index('abc') #return the index



# PATTERNS :

for i in range(1,4):
 print('Student',i)
 for j in range(1,6):
  print('Subject',j,'for student',i)
 print('###### TEA BREAK #######')



for i in range(1,4):
 print('Student',i)
 for j in range(1,6):
  print('Subject',j,'for student',i,end=' ')
 print('###### TEA BREAK #######')


for i in range(1,4):
 print('Student',i)
 for j in range(1,i+1):
  print('Subject',j,'for student',i,end=' ')
 print(' ')


for i in range(1,4):
 for j in range(1,4):
  print('*',end=' ')
 print(' ')

.


for i in range(1,4):
 for j in range(1,4-i):
  print('*',end=' ')
 print(' ')

# Questions:
for i in range(1,5):
 for j in range(1,5):
  print(i,end=' ')
 print(' ')


for i in range(2,5):
 for j in range(1,5):
  print(i*2,end=' ')
 print(' ')


for i in range(1,4):
 for j in range(1,5):
  print(j,end=' ')
 print(' ')



for i in range(4,8):
 for j in range(1,2):
  print('4567',end=' ')
 print(' ')


for i in range(1,13):
 for j in range(1,i+1):
  print(i,end=' ')
 print(' ')


for i in range(1,5):
 for j in range(1,i+1):
  print('*',end=' ')
 print(' ')



for i in range(1,5):
 for j in range(1,i+1):
  print(j,end=' ')
 print(' ')


for i in range(4,8):
 for j in range(4,i+1):
  print(j,end=' ')
 print(' ')



k=1
for i in range(1,5):
 for j in range(1,i+1):
  print(k,end=' ')
  k+=1
 print(' ')



for i in range(65,70):
 for j in range(65,i+1):
  print(chr(i),end=' ')
 print(' ')



for i in range(1,5):
 for j in range(1,5-i):
  print('-',end=' ')
 for j in range(1,i+1):
  print('*',end=' ')
 print(' ')



for i in range(1,6):
 for j in range(2,i+1):
  print('-',end=' ')
 for j in range(0,6-i):
  print('*',end=' ')
 print(' ')



for i in range(1,5):
 for j in range(1,i+1):
  if((i+j)%2==0):
   print('1',end=' ')
  else:
   print('0',end=' ')
 print(' ')



-----------------------------------------------------------------------------------

15.05.2025

# DICTIONARY : dataype
# Key-value
# key= unique identifier
# value: unique/duplicate
# dictionary is mutable
# no index position

mydictionary={10:'Enosh', 20:'James'}
print(mydictionary[10])

mydictionary[10]='Arthur' #update the value
mydictionary['Amount']=1000  #insert the value

print(mydictionary)
mydictionary.pop(20)  # delete using pop function and key name
print(mydictionary)

print(help(mydictionary))  #documentation which tells all the functions of the datatype you are working in
 

mydictionary.keys()
mydictionary.values()


# counting total characters

data='hello'     
count=0
for char in data:
 count+=1
mydictionary={'total',count}
print(mydictionary)



data='user'
mydictionary={}
for char in data:
  mydictionary[char]=1
print(mydictionary)



data='hey isha'
mydictionary={}
for char in data:
 if char in 'aeiou':
  mydictionary[char]=1
print(mydictionary)



data='hey ishaeeaeiouaeiou'
mydictionary={}
for char in data:
 if char in 'aeiou':
  if char not in mydictionary:
   mydictionary[char]=1
  else:
   mydictionary[char]=mydictionary[char]+1
print(mydictionary)



# LIST COMPREHENSION:

[i+5 for i in [10,20,30,40,50]]



# DICTIONARY COMPREHENSION:

{char:1 for char in 'hello'}



# basic loops, if-else, list, dictionary, tuple
# function: set of statement=> task to perform
# logic => code reusability

a=100  #global variable that means it is accessible outside the function
def test():
 z=19  #local variable that means it is not accessible outside the function
 print('Hello',z,a)


print(a)



def msg(username): #parameter
 print('Hello User',username)



def totalSum(num):
 total=0
 for i in range(1,num+1):
  total=total+i
 print(f'Sum of {num} is : ',total)



for i in range(1,5):
 for j in range(2,i+1):
  print('-',end='')
 for j in range(1,6-i):
  print(j,end='')
 print('')



def pattern(n):
 for i in range(1,n):
  for j in range(2,i+1):
   print('-',end='')
  for j in range(1,n-i+1):
   print(j,end='')
 print('')


# function a
def func(a,b):
 print('a: ',a, 'b: ',b)


def func():
 print('hello')



x=func  #first class function
func()
x()



first class functions vs high order functions


-----------------------------------------------------------------------------------



16.05.2025

# LAMBDA FUNCTION
# one line function
# anonymous function
# they don't have a name

y= lambda num: num+5
y(10)


# MAP FUNCTION
# It applies on each and every element and return a value
# one element is loaded in the memory and processed at a time

list(map(len,['hey','hello']))
tuple(map(len,['enosh','james']))

list(map(lambda x : x*x,[10,20,30,40,50]))
list(filter(lambda x : x%2==0,[10,20,30,40,50]))


-----------------------------------------------------------------------------------


17.05.2025

# OOPS IN PYTHON

class HouseDesign:
 color='yellow'

#object=class()
h1=HouseDesign()
print(h1.color)


h2=HouseDesign()
h2.color='White'
print(h2.color)


class HouseDesign:
 def __init__(self):  #self => current object reference is stored 
  print('Worker has arrived')


h1.HouseDesign()



class HouseDesign:
 def __init__(self,x):
  self.color=x         #h1.color=green


h1=HouseDesign('Green')
print('h1 : ',h1.color)

h2=HouseDesign('Red')
print('h2 : ',h2.color)


# INHERITANCE => Parent class => child class
# this makes code reusable

class Parent:
 amount=5000

class child(Parent):
 salary=10000

c1=child()
print(c1.amount)




class Driver:
 def __init__(self,name,id,email):
  self.name=name
  self.id=id
  self.email=email

x1=Driver('Enosh','048','jamesenosh@gmail.com')
x2=Driver('Hardik','054','hardik@gmail.com')

print('Name : ',x1.name, '\nID : ',x1.id, '\nEmail : ',x1.email)
print('Name : ',x2.name, '\nID : ',x2.id, '\nEmail : ',x2.email)




class customer:
 def __init__(self,id,name,email,wallet):
  self.id=id
  self.name=name
  self.email=email
  self.wallet=wallet

a1=customer('01','Enosh','enosh@gmail.com','10000')
a2=customer('02','Hardik','hardik@gmail.com','20000')

print('ID : ',a1.id, '\nName : ',a1.name, '\nEmail : ',a1.email, '\nWallet : ',a1.wallet)





class customer(Driver): #using inheritance here
 def __init__(self,id,name,email):
  super().__init__(id,name,email)
 

a1=customer('01','Enosh','enosh@gmail.com')
print('ID : ',a1.id, '\nName : ',a1.name, '\nEmail : ',a1.email)


class Employee:
 def __init__(self,a,b,c,d):
  self.id=a
  self.name=b
  self.email=c
  self.salary=d

 def info(self):
  print(self.salary//12, self.email.split('@')[-1])


a1=Employee('01','Enosh','enosh@gmail.com','800000')

print('ID : ',a1.id, '\nName : ',a1.name, '\nDomain : ',a1., '\nMonthly Salary : ',a1.)


----------------------------------------------------------------------------------------------------------


19.05.2025


# DATA SCIENCE
# Problem => Data acquire => Data pre-process => data storage
# Data preparation => Data analysis => Data visualize
# ML and Algorithmn ( Chatbot, Open AI service, Open cv, NLP) => Data report

# Python => libraries => Pandas and numpy
# Numpy => numerical python
# Scientific library => data in form of nd array
# [  1,2,3,4,  ]
#   [1,2],'
#   [3,4]
# Data prepare in form of number

import numpy
numpy.arrange(7)
type(numpy.arrange(7))

import numpy as np
arr1=np.arange(5)
arr1

np.ndarray(3)

arr1=np.array([1,2,3])
arr1.size

arr1=np.array([ [1,2],[3,4],[5,6] ])
print(arr1.size)
print(arr1.shape)
print(arr1.dtype)

arr2=np.arange(3)
print(arr2.shape)

np.ones([3,4])

arr1=np.arange(9)
arr1.reshape(3,3)

import numpy as np
a=np.array([ [1,2],[3,4] ])
print(a)
b=a.transpose()
print(b)


arr1=np.array([[[1.0,2],[2,3],[5,6]]])
arr1.ndim
arr2.shape


# NUMPY
# String => convert => int


# pandas : powerful => data transform, clean, prepare
# Series : 1D array
# Dataframe : collection of series tabular format for your data 

import pandas as pd
pd.Series([10,2,34,4])


import pandas as pd
series1=pd.Series([10,2,34,4])
series1.values
series1.index
print(series1[0])
series1[0]=90      #update
series[5]=1000
series1

series1.max()
series1.idxmax()

series1.max()
print(series1.idxmin())


import pandas as pd
series1=pd.Series([10,2,34,4], index=['A','B','C','D']) #series with customize index
series1.to_dict()

arr1=np.array([10,13,15,19,24])
series1=pd.Series(arr1)
series1
type(series1)


arr1=np.array([10,13,15,19,24])
series1=pd.Series(arr1)
series1.value_counts()


arr1=np.array([10,13,15,19,24])
series1=pd.Series(arr1)
series1.value_counts(ascending=True)  #ascending is false by default



arr1=np.array([10,13,15,19,24])
series1=pd.Series(arr1)
series1.is_unique  #do we have unique value or not

series1.nunique()  # total unoque element


arr1=np.array([10,13,15,19,24,10,15])
series1=pd.Series(arr1)
series1.drop_duplicates()  # removes duplicate values but original values are not affected
series1.drop_duplicates(inplace=True)  # removes duplicate values in the original series also





arr1=np.array([10,13,15,19,24,10,15])
series1=pd.Series(arr1)
series1.values
series1.index
series1.max()
series1.idxmax()
series1.min()
series1.idxmin()
series1.value_counts()
series1.value_counts(ascending=True)  
series1.drop_duplicates()  
series1.drop_duplicates(inplace=True)
series1.is_unique()
series1.nunique()
series1.to_dict()


data=[ [10,12],[13,14] ]
pd.DataFrame(data)

df=pd.DataFrame(data, columns=['Product1','Product2','Product3'])
type(df['Product1'])


import os
import pandas as pd
df=pd.read_csv('D:\\enosh_regex\\pyprogs\\ml-latest-small\\ratings.csv')
print(df)



--------------------------------------------------------------------------------------------



20.05.2025

# INTRODUTION TO PANDAS : is anopen sourcce library that is used to handle data manipulations.

# 1.) Series
# 2.) DataFrame


# 1.) Series : It is an one dimensional array and it shows omly values not column name.

a=pd.Series([1,233,67,90])
print(a)
type(a)


# 2.) DataFrame : It is a multi dimensiona; array and it has values with column name.

a={ 'Name' : ['Sam','Raj','Rahul','Gaurav'],
    'Domain' : ['D.E','D.S','D.S','Full Stack'],
    'Duration' : [30,15,45,30]
  }

df=pd.DataFrame(a)
print(df)



df=pd.read_csv("D:\\enosh_regex\\pyprogs\\netflix_titles.csv")
print(df)


df=pd.read_csv("D:\\enosh_regex\\pyprogs\\makemytrip.csv")
print(df)

-----------------------------------------------------------------------------------

21.05.2025

import pandas as pd 
df=pd.read_csv('D:\\enosh_regex\\csv_file_work\\titanic.csv')
print(df)

print(df.tail())
print(df.head())
print(df.sample(3))
print(df.shape)
print(df.columns)

print(df.loc[2:5, ['Survived','Pclass'] ])    #df.loc[low_ramge,column_name]
print(df.iloc[ [] ])
 

print(df.dtypes)
print(df.isnull().sum())   #returns total missing data in each column.

p=df.dropna()  #remove missing rows.
print(p.isnull().sum())

print(df.drop(columns=['Cabin']))
print(df.sample(3))

df['Fare']=df['Fare'].fillna(5)
df['Age']=df['Age'].fillna(10)
print(df.head())

df['Fare']=df['Fare'].astype(int)
print(df['Fare'].dtype)

df['Survived'].value_counts()   

df.rename(columns={'Age':'Updated_Age'})
print(df.head(3))


# DATA
# 1.) Continous data => we can divide in more subdata eg dob,river_length
# 2.) Discrete data => we cannot divide in more sub_data
# Ex. Marital status, total_number_of _employees_in_a_company


# Exploratory Data Analysis
# 1.) Univariate Analysis : Analysis on a single column
# 2.) Bivariate Analysis : Analysis on 2 columns
# 3.) Multivariate Analysis : Analysis on more than 2 columns

# Numerical : Histogram, line chart
# Categorical : pie chart, bar chart, countplot.

import matplotlib as 
import seaborn as sns

print(df.columns)

print(sns.countplot(x=df['Survived']))

df['Survived].value_counts().plot(kind='bar')
df['Survived].value_counts().plot(kind='pie',autopct='%.2f')

plt.hist(x=df['Updated_Age'])
plt.show()

sns.boxplot(x=df['Survived'])

sns.scatterplot(x=df['total_bill'], y=df['tip'])
sns.scatterplot(x='total_bill', y=df'tip', data=df, hue=df['sex'])

sns.scatterplot(x='total_bill', y=df'tip', data=df, hue=df['sex'],style=df['smoker'])

# Heatmap(Categorical-Categorical)
p=pd.crosstab(df['Day'],df['time'])
print(p)
sns.heatmap(p)

df.groupby('time').sum()['total_bill']

-----------------------------------------------------------------------------------

22.05.2025

# SUPPLY CHAIN PROJECT

import numpy as np 
import pandas as pd

df=pd.read_csv('D:\\enosh_regex\\Datasets\\supply_chain.csv')


#print(df.head())


import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
pio.templates.default='plotly_white'


df.describe()


fig=px.scatter(df,x='Price', y='Revenue generated', color='Product type', hover_data=['Number of products sold'], trendline = 'ols')
fig.show()


sales_data=df.groupby('Product type')['Number of products sold'].sum().reset_index()

pie_chart=px.pie(sales_data, values='Number of products sold', names='Product types', title='Sales by product type', hover_data=['Number of product sold'], hole=0.5, color_discrete_sequence = px.colors.qualitative.Pastel)

pie_chart_update_traces(textposition='inside', textinfo='percent+label')
pie_chart.show()


total_revenue=df.groupby('Shipping carriers')['Revenue generated'].sum().reset_index()

fig=go.Figure()

fig.add_trace(go.bar(x=total_revenue['Shipping carriers'], y=total_revenue['Revenue generated']))

fig.update_layout(titlt='Toatl Revenue by Shipping Carrier', xaxis_title='Shipping Carrier', yaxis_title='Revenue Generated')

fig.show()



avg_lead_time=df.groupby('Product type')['Lead time'].mean().reset.index

avg_manufacturing_costs + df.groupby('Product type ')['Manufacturing costs'].mean().reset_index()

result = pd.merge(avg_lead_time, avg_manufacturing_costs, on='Product type')

result.rename(columns={'Lead time': 'Average Lead Time', 'Manufacturing costs' : 'Average Manufacturing Costs'}, inplace=True)

print(result)


revenue_chart=px.line(df, x='SKU', y='Revenue generated', title='Revenue generated ', by SKU)
revenue_chart.show()

stock_chart = px.bar(x='SKU', y='Stock levels', title='Stock Levels by SKU')
order_quantity_chart.show()

order_quantity_chart = px.bar(x='SKU', y='Order quantities', title='Order Quantity by SKU')
order_quantity_chart.show()

shipping_cost_chart=px.bar(df,x='Shipping carriers', y='Shipping costs', titlt='Shipping Costs by Carriers')
shiping_cost_chart_.show()

transportation_chart=px.pie(df, values='Costs', names='Transportation modes', title='Cost Distribution by Transportation Mode', hole=0.5, color_discrete_sequence=px.colors.qualitative.Pastel)
transportation_chart.show()

'''


# LINKEDIN REVIEWS PROJECT

import numpy as np 
import pandas as pd

df=pd.read_csv('D:\\enosh_regex\\Datasets\\linkedin-reviews.csv')

import matplotlib.pyplot as plt
import seaborn as sns

'''
sns.set(style='whitegrid')
plt.figure(figsize=(9,5))
sns.countplot(data=df, x='Rating')
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()
'''

from textblob import TextBlob
def textblob_sentiment_analysis(review):
 sentiment=TextBlob(review).sentiment
 if(sentiment.polarity>0.1):
  return 'Positive'
 elif(sentiment.polarity<0.1): 
  return 'Negative'
 else:
  return 'Neutral'

df['Sentiment']=df['Review'].apply(textblob_sentiment_analysis)
df.sample(5)

sentiment_distribution = df['Sentiment'].value_counts()
sentiment_distribution

--------------------------------------------------------------------------------------------------------------

24.05.2025


#Introduction to Machine Learning------>

#Data ---> divide (Input, target) -----> trained by ML Model ----> ML Model data Prediction -----> 

#Model Accuracy

#Normal Distribution ---->
-3 -2 -1 0 1 2 3

#here 0 ---> Central tendancy(mean, median)
#gap(Standard deviation) equal

#Why we convert our data into Normal Distribution?
#(1) Calculation Easy
#(2) take less time for training and executing

# ML -----> Feature Engineering (Create Appropriate data) + Solution

#Feature Engineering

#(1) Data Dividation


import pandas as pd
df=pd.read_csv('D:\\enosh_regex\\Datasets\\covid_toy.csv')

df.head(3)


s1=SimpleImputer() # it will fill missing values

# if data
# Numerical ---> fill data mean
# Categorical ---> fill data most frequent values

df['fever'] = s1.fit_transform

df['gender'].value_counts()  # it will return total_frequency

df['cough']=df['cough'],map({'Mild':0})


















































