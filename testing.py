import os

for dirPath, _, fileList in os.walk('./CGED'):
	for fileName in fileList:
		if 'cbe' in fileName: continue
		fullName = os.path.join(dirPath, fileName)
		maxlen = 0
		maxline = ''
		with open(fullName, 'r', encoding='UTF-8') as finn:
			for line in finn:
				if len(line.strip()) > maxlen:
					# print(len(line.strip()), line, end='')
					maxlen = len(line.strip())
					maxline = line
		print(maxlen, fileName, maxline, end='')

n = 16384

for i in range(1, n):
	if n % i == 0: print('%5d x %5d' % (i, int(n/i)))

# fin1 = open('./CGED/char_level/cbe_train_x.txt', 'r', encoding='UTF-8')
# fin2 = open('./CGED/char_level/cbe_train_y.txt', 'r', encoding='UTF-8')

# for line1 in fin1:
# 	line2 = fin2.readline()
# 	if len(line1) != len(line2):
# 		print(len(line1), len(line2), line1, line2)
# 		break

# fin1.close()
# fin2.close()
