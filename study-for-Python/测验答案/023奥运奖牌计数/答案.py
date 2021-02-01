n = int(input())
ls = [0, 0, 0]
for i in range(n):
    s = list(map(int, input().split()))
    for j in range(3):
        ls[j] += s[j]
for i in range(3):
    print(ls[i], end=' ')
print(sum(ls))