from sklearn.metrics import mean_squared_error
import math
import numpy as np

a = [1, 2, 3]
b = [1, 1, 1]

print math.sqrt(mean_squared_error(a, b))

l = 0
for i in xrange(len(a)):
    l += (a[i] - b[i])**2 / 3.0
print math.sqrt(l)


a = [1,2,4,5,6,3]

print np.linalg.norm(a)**2