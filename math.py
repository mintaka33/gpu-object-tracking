import numpy as np

def cos2d():
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

w, h = 20, 10
sigma = 2.0

xs, ys = np.meshgrid(np.arange(w), np.arange(h))
center_x, center_y = w / 2, h / 2
dist = ((xs - center_x) ** 2 + (ys - center_y) ** 2) / (sigma**2)
labels = np.exp(-0.5*dist)

np.savetxt('dist.csv', dist, delimiter=',')
np.savetxt('labels.csv', labels, delimiter=',')

print('\ndone')
