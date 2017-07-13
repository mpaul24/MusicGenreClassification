import numpy as np

a = np.ones([1, 2])
b = np.ones([2, 2])
c = np.ones([2, 1])
#print(c)
#print(a, b, c)
print(np.dot(np.dot(a, b), c))