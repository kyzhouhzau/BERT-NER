import random

ain = open('./a.txt', 'r', encoding='UTF-8')
a = ain.readlines()
ain.close()

random.shuffle(a)

n = int(len(a) / 10)

# test
with open('./test.txt', 'w', encoding='UTF-8') as test:
	test.writelines(a[:n])

# dev
with open('./dev.txt', 'w', encoding='UTF-8') as dev:
	dev.writelines(a[n:2*n])

# train
with open('./train.txt', 'w', encoding='UTF-8') as train:
	train.writelines(a[2*n:])
