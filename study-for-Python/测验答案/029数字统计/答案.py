s = list(map(int, input().split()))
cnt = 0
for i in range(s[0], s[1]+1):
    for j in str(i):
        if j == '2':
            cnt += 1
print(cnt)