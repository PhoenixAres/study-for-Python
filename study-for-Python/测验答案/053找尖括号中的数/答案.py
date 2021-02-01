import re
t = int(input())
m = r'<([1-9]\d{0,2}|0)>'
for i in range(t):
    s = input()
    ls = re.findall(m, s)
    if ls == []:
        print('NONE')
        continue
    print(' '.join(ls))