with open('./a.txt', 'r', encoding='UTF-8') as fin:
	for i, a in enumerate(fin):
		print(i, a)