n = int(input())
ls = []
for i in range(n):
    ls.append(list(input().split()) + [i])
a = list(filter(lambda x:int(x[1]) >= 60, ls))
b = list(filter(lambda x:int(x[1]) < 60, ls))
a.sort(key=lambda x:(-int(x[1]), x[2]))
b.sort(key=lambda x:x[2])
for i in a:
    print(i[0])
for i in b:
    print(i[0])