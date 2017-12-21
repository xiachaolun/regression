from sklearn.metrics import mean_squared_error

a = [1, 2, 3]
b = [1, 1, 1]

print mean_squared_error(a, b)

l = 0
for i in xrange(len(a)):
    l += (a[i] - b[i])**2 / 3.0
print l