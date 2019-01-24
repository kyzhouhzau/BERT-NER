with open('./bert.txt', 'r', encoding='UTF-8') as fin:
    data = fin.readlines()

i = 0
ds = []
while i < len(data):
    x = data[i+1].strip()
    y = data[i+2].strip()
    ds.append('%s\t%s\n' % (x, y))
    i += 3

fold = int(len(ds) / 10)
with open('test.txt', 'w', encoding='UTF-8') as fout:
    for i in range(fold):
        fout.write(ds[i])

with open('dev.txt', 'w', encoding='UTF-8') as fout:
    for i in range(fold, fold*2):
        fout.write(ds[i])

with open('train.txt', 'w', encoding='UTF-8') as fout:
    for i in range(fold*2, len(ds)):
        fout.write(ds[i])