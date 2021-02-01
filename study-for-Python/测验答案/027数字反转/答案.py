s = input()
print('-' + str(int(s[1:][::-1])) if s[0] == '-' else str(int(s[::-1])))