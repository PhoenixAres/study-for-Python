s = list(map(int, input().split()))
a = list(map(int, input().split()))
b = list(map(int, input().split()))
cnt = [0, 0]
for i in range(s[0]):
    if a[i%s[1]] - b[i%s[2]] in [-2, 5, -3]:
        cnt[0] += 1
    elif a[i%s[1]] - b[i%s[2]] in [-5, 2, 3]:
        cnt[1] += 1
if cnt[0] > cnt[1]:
    print('A')
elif cnt[0] < cnt[1]:
    print('B')
else:
    print('draw')