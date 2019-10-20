import csv

data=list(csv.reader(open('data_1.csv','r')))
print('[%, %, %, %, %, %]')
h=data[0][:-1]
print(h)
for row in data[1:]:
    if row[-1]=='Y':
        for i in range(len(row[:-1])):
            if h[i]=='?':
                continue
            elif h[i]!=row[i]:
                h[i]='?'
        print(h)


# OUTPUT:
# [%, %, %, %, %, %]
# ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same']
# ['Sunny', 'Warm', '?', 'Strong', 'Warm', 'Same']
# ['Sunny', 'Warm', '?', 'Strong', '?', '?']