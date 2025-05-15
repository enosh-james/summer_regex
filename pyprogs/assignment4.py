'''
# 1.)

for i in range(4):
 print(' ' * i + '*' * 4)



# 2.)

for i in range(5):
 print(' '*i,end='')
 for j in range(1,6-i):
  print(j,end='')
 print()



# 3.)

for i in range(5,0,-1):
 print(' '*(i-1)+'1 2 3 4 5')
 
'''

# 4.)


for i in range(1,5):
 for j in range(1,6-i):
  print('',end=' ')
 for j in range(1,5):
  print(j,end=' ')
 print('')



















