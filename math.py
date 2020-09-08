import numpy as np
from numpy.fft import fft
import cmath

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

def guassian2d():
    w, h = 100, 100
    sigma = 2.0

    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    center_x, center_y = w / 2, h / 2
    dist = ((xs - center_x) ** 2 + (ys - center_y) ** 2) / (sigma**2)
    labels = np.exp(-0.5*dist)

    np.savetxt('e:\\xs.csv', xs, delimiter=',')
    np.savetxt('e:\\ys.csv', ys, delimiter=',')
    np.savetxt('e:\\dist.csv', dist, delimiter=',')
    np.savetxt('e:\\labels.csv', labels, delimiter=',')

def dft_fft():
    a = [11.0, 22.0, 33.0, 44.0]
    b = fft(np.array(a))
    print(b)
    X = [abs(f) for f in b]
    print(X)

    N = 4
    c = -2*np.pi/N
    W0 = cmath.exp(complex(0, 0*c))
    W1 = cmath.exp(complex(0, 1*c))
    W2 = cmath.exp(complex(0, 2*c))
    W3 = cmath.exp(complex(0, 3*c))
    W4 = cmath.exp(complex(0, 4*c))
    W6 = cmath.exp(complex(0, 6*c))
    W9 = cmath.exp(complex(0, 9*c))

    print(W0, W1, W2, W3, W4, W6, W9, sep='\n')
    print([abs(n) for n in [W0, W1, W2, W3, W4, W6, W9]])
    print(cmath.exp(complex(0, -2*np.pi)))
    W = np.array([[W0, W0, W0, W0], [W0, W1, W2, W3], [W0, W2, W4, W6], [W0, W3, W6, W9]])
    X = np.array(a)
    F = np.dot(W, X)
    print(F)
    print([abs(f) for f in F])

print('\ndone')
