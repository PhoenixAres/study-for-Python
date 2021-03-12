d = ("Sunday","Monday","Tuesday","Wednesday","Thursday", "Friday","Saturday")
m = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30 ,31]
t = int(input())
for i in range(t):
    s = list(map(int, input().split()))
    m[1] = 29 if s[0]%4 == 0 and s[0]%100 != 0 or s[0]%400 == 0 else 28
    if s[1] < 1 or s[1] > 12:
        print('Illegal')
        continue
    if s[2] < 1 or s[2] > m[s[1]-1]:
        print('Illegal')
        continue
    day = 0
    if s[0] <= 2020:
        for j in range(1, s[1]):
            day -= m[j-1]
        day -= s[2]
        for j in range(s[0], 2020):
            day += 366 if j%4 == 0 and j%100 != 0 or j%400 == 0 else 365
        m[1] = 29
        for j in range(1, 11):
            day += m[j-1]
        day += 18
        print(d[(10-day%7)%7])
    else:
        for j in range(1, s[1]):
            day += m[j-1]
        day += s[2]
        for j in range(2020, s[0]):
            day += 366 if j%4 == 0 and j%100 != 0 or j%400 == 0 else 365
        m[1] = 29
        for j in range(1, 11):
            day -= m[j-1]
        day -= 18
        print(d[(day%7+3)%7])