import numpy as np

sz = [20, 10]

a = np.hanning(int(sz[1]))
b = np.hanning(int(sz[0]))

x = a[:, np.newaxis]
y = b[np.newaxis, :]

w = x.dot(y)

for y in range(sz[1]):
    print('\n', end='')
    for x in range(sz[0]):
        print('%f' % w[y][x], end=', ')

print('\ndone')
