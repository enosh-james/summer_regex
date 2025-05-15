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

'''

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

















































