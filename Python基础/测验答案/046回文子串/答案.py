s = input()
n = len(s)
ans = []
for i in range(2, n+1):
    for j in range(n):
        if j + i > n:
            continue
        if s[j:j+i] == s[j:j+i][::-1]:
            ans.append(s[j:j+i])
for i in ans:
    print(i)