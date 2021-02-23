n, m = map(int, input().split())
d = {}
for i in range(m):
    s = input().split()
    d[s[0]] = [int(s[1]), int(s[2])]
ans = 0
for i in range(n):
    s = input().split()
    for j in s:
        if d[j][1] > 0:
            d[j][1] -= 1
            ans += d[j][0]
print(ans)