import re
t = int(input())
for i in range(t):
    ls  = re.findall(r'(<([a-z]+)>)(.*?)(</\2>)', input())
    flag = True
    for x in ls:
        lst = re.findall(r'\((\d{1,2})\)-(\d+)', x[2])
        ans = [j[0] for j in lst if len(j[1]) == 3]
        if ans != []:
            print(x[0] + ','.join(ans) + x[3])
            flag = False
    if flag:
        print('NONE')