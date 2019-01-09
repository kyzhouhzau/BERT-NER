xout = open('./x.txt', 'w', encoding='UTF-8')
yout = open('./y.txt', 'w', encoding='UTF-8')
aout = open('./a.txt', 'w', encoding='UTF-8')

with open('./CGED_Identification.txt', 'r', encoding='UTF-8') as fin:
	s1, s2 = str(), str()
	i = 0
	for line in fin:
		i += 1
		if line == '\n': continue
		try:
			a, b = line.strip().split('\t')
			if a == 'end' or a == '，' or a == '。' or a == '！':
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