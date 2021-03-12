t = int(input())
for i in range(t):
    a, b = input().split()
    c = a.find(b)
    if c == -1:
        print('no')
        continue
    while c != -1:
        print(c, end=' ')
        c = a.find(b, c+len(b))
    print()