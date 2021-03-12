n = int(input())
ls = []
for i in range(n):
    ls.append(int(input()))
print(sum(ls), '%.5f' % (sum(ls)/n))