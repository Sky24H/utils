import os
import shutil

# compare files in the two directory
a = './a/'
b = './b/'


list_a = os.listdir(a)
list_b = os.listdir(b)
count = 0

for filename in list_b:
    if filename in list_a:
        # add process
        count += 1
print(count, '/'+len(list_b))
