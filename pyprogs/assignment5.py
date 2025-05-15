'''

# 1.)


def lcm(a,b):
 a=int(a)
 b=int(b)
 if(a>b):
  greater=a
 else:
  greater=b
 print(greater)
 while(True):
  if(greater%a==0 and greater%b==0):
   print('lcm : ',greater)
   break
  else:
   greater+=1



# 2.)

def is_armstrong(num):
 num_str=str(num)
 num_digits=len(num_str)
 total=sum(int(digit) ** num_digits for digit in num_str)
 return total==num

'''

# 3.)

 
def is_palindrome(num):
 num=str(num)
 start=0
 end=len(num)-1
 a=0
 while(start<end):
  if(num[start] != num[end]):
   a=1            
   break
  start+=1
  end-=1
 if(a==1):
  print('Not a Palindrome')
 else:
  print('Palindrome')
  














































