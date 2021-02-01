n = int(input())
ans = 0
sum = 0
for i in range(n):
    s = list(map(int, input().split()))
    if 90 <= s[0] <= 140 and 60 <= s[1] <= 90:
        sum += 1
    else:
        ans = max(ans, sum)
        sum = 0
ans = max(ans, sum)
print(ans)