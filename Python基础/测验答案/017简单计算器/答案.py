s = input().split()
if s[2] not in '+-*/':
    print('Invalid operator!')
elif s[2] == '/' and s[1] == '0':
    print('Divided by zero!')
else:
    print(int(eval(s[0] + s[2] + s[1])))