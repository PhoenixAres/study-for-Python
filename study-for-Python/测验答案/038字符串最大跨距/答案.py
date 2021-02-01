s = input().split(',')
a, b = s[0].find(s[1]), s[0].rfind(s[2])
print(b-a-len(s[1]) if a != -1 and b != -1 and a+len(s[1]) <= b else -1)