s = list(map(int, input().split()))
s.sort()
print('yes' if s[0]+s[1] > s[2] else 'no')