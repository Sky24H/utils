import os
import shutil

a = './a/'
b = './b/'
bx = './x/'

list_a = os.listdir(a)
list_b = os.listdir(b)
count = 0

for filename in list_b:
    if filename in list_a:
        # print('ok')
        #shutil.copy(b+filename, x+filename)
        count += 1
print(count, '/'+len(list_b))
