s = input()
for i in s:
    if 'a' <= i <= 'z':
        print(chr(ord(i)-32), end='')
    elif 'A' <= i <= 'Z':
        print(chr(ord(i)+32), end='')
    else:
        print(i, end='')