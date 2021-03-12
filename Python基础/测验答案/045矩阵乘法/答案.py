n, m, q = list(map(int, input().split()))
a = []
for i in range(n):
    a.append(list(map(int, input().split())))
b = []
for i in range(m):
    b.append(list(map(int, input().split())))
c = [[0 for i in range(q)] for j in range(n)]
for i in range(n):
    for j in range(m):
        for k in range(q):
            c[i][k] += a[i][j]*b[j][k]
for i in range(n):
    for j in range(q):
        print(c[i][j], end=' ')
    print()