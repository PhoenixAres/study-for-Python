from math import ceil
s = list(map(int, input().split()))
print(max(0, s[0]-ceil(s[2]/s[1])))