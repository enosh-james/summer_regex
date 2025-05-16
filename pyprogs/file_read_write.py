'''

f= open('user.txt','w')
# print(f.read())
f.write('Enosh Arthur James')
#print(f.read())
f.close()

#open => read/write => close



f= open('abc.txt','w')
f.write('Enosh Arthur James')
f.close()

#open => read/write => close


f= open('abc.txt','r+')
print(f.read())
f.write('India')
f.close()


import csv

with open('Global_AI_Content_Impact_Dataset.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        print(f"Column 1: {row[0]}, Column 2: {row[1]}")

import csv

data = [
    ['Name', 'Age', 'Department'],
    ['John Doe', 32, 'Marketing'],
    ['Jane Smith', 28, 'HR'],
    ['Alex Johnson', 40, 'Finance']
]
with open('Global_AI_Content_Impact_Dataset.csv', 'w', newline='') as file:
 writer = csv.writer(file)
 writer.writerows(data)



import csv
f=open('abc.txt'.'r')
data=csv.reader(f)
for line in data:
 print(line)

'''




# USING A PYTHON PROGRAM READ THE TABLE PRESENT IN MYSQL DATABASE AND SAVE THE CONTENT INTO THE FILE ON DESKTOP.





































