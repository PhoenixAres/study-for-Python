def dfs(n, m):
    if n == 0:
        return 1
    num = 0
    for i in range(m, n+1):
        num += dfs(n-i, i)
    return num

n = int(input())
print(dfs(n, 1))