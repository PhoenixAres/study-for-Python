from math import sqrt
a, b, c = map(float, input().split())
b = -b if b == 0 else b
if b**2-4*a*c > 0:
    print('x1=%.5f' % ((-b + sqrt(b*b-4*a*c))/(2*a)) + ';x2=%.5f' % ((-b - sqrt(b*b-4*a*c))/(2*a)))
elif b**2-4*a*c == 0:
    print('x1=x2=%.5f' % (-b/(2*a)))
else:
    print('x1=%.5f%+.5fi' % (-b / (2*a), sqrt(4*a*c-b*b) / (2*a)) + ';x2=%.5f%+.5fi' % (-b / (2*a), -sqrt(4*a*c-b*b) / (2*a)))