from copy import deepcopy
n, m = map(int, input().split())
a = []
for i in range(n):
    a.append(list(map(int, input().split())))
b = deepcopy(a)
for i in range(1, n-1):
    for j in range(1, m-1):
        b[i][j] = round((a[i][j] + a[i-1][j] + a[i+1][j] + a[i][j-1] + a[i][j+1])/5)
for i in range(n):
    for j in range(m):
        print(b[i][j], end=' ')
    print()