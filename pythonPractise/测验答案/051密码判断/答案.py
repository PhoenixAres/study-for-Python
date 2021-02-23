import re
m = '[a-zA-Z][a-zA-Z0-9-_]{7,}$'
while True:
    try:
        s = input()
        if re.match(m,s) != None:
            print("yes")
        else:
            print("no")
    except:
        break