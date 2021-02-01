n = int(input())
s = list(map(float, input().split()))
x = s[1]/s[0]*100
for i in range(n-1):
    s = list(map(float, input().split()))
    y = s[1]/s[0]*100
    if y-x > 5:
        print('better')
    elif x-y > 5:
        print('worse')
    else:
        print('same')