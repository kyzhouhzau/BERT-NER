xout = open('./x.txt', 'w', encoding='UTF-8')
yout = open('./y.txt', 'w', encoding='UTF-8')
aout = open('./a.txt', 'w', encoding='UTF-8')
split_tag = ['end', '，', '。', '！', '、', '「', '」', '（', '）', '？', '；']

def isin(s, t):
	for tt in t:
		if tt == s: return True
	return False

with open('./CGED_Identification.txt', 'r', encoding='UTF-8') as fin:
	s1, s2 = str(), str()
	i = 0
	for line in fin:
		i += 1
		if line == '\n': continue
		try:
			a, b = line.strip().split('\t')
			if isin(a, split_tag):
				if s1.strip() != '':
					xout.write(s1 + '\n')
					yout.write(s2 + '\n')
					aout.write('%s\t%s\n' % (s1, s2))
				if len(s1) != len(s2):
					print(len(s1), len(s2))
					print(i, s1, s2)
				s1, s2 = str(), str()
			else:
				s1 += a[0]
				s2 += b[0]
		except: print(i, 'Not enough')

xout.close()
yout.close()

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
