import os
a = './results/'
b = './datasets/flickr/'
count = 0
list_a = []
for root,dirs,files in os.walk(a):
	for name in files:
		list_a.append(name[:len(name)-4])

for i in range(10):
	for root,dirs,fff in os.walk(b+str(i)+"/"):
		for name in fff:
			if name[:len(name)-4] in list_a:
				s = b+str(i)+"/"+name
				d = "./done/"+name
				os.rename(s,d)
				print('ok')
				count += 1
			else:
				print("ng")
print(count)
