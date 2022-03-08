import random

lines = open('c3_test.labels').readlines()

out = open('c3_test.list', 'w')
for line in lines:
    newline = line.rsplit(' ', 1)[0]
    print(newline)
    out.write(newline+'\n')
