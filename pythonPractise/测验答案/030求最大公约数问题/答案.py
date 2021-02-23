s = list(map(int, input().split()))
t = s[0] % s[1]
while t:
    s[0] = s[1]
    s[1] = t
    t = s[0] % s[1]
print(s[1])