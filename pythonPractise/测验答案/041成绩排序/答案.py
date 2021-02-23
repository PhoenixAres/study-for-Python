t = int(input())
ls = []
for i in range(t):
    ls.append(input().split())
ls.sort(key=lambda x:(-int(x[1]), x[0]))
for i in ls:
    print(i[0], i[1])