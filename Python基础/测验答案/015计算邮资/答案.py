from math import ceil
s = input().split()
a, b = int(s[0]), s[1]
s = (ceil((a-1000)/500))*4+8 if a > 1000 else 8
print(s if b == 'n' else s+5)