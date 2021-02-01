def ways(m, n, s):
    if n == 0 and s == 0:
        return 1
    if m == 0 or n == 0 or s == 0:
        return 0
    w = ways(m-1, n, s)
    if s >= m:
        w += ways(m-1, n-1, s-m)
    return w

t = int(input())
for i in range(t):
    s = list(map(int, input().split()))
    print(ways(s[0], s[1], s[2]))