import numpy as np
from numpy.fft import fft
import cmath
from cmath import exp, pi

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

def dft():
    a = [1.0, 2.0, 3.0, 4.0]
    N = 4
    c = -2*np.pi/N
    W0 = cmath.exp(complex(0, 0*c))
    W1 = cmath.exp(complex(0, 1*c))
    W2 = cmath.exp(complex(0, 2*c))
    W3 = cmath.exp(complex(0, 3*c))
    W4 = cmath.exp(complex(0, 4*c))
    W6 = cmath.exp(complex(0, 6*c))
    W9 = cmath.exp(complex(0, 9*c))

    #print(W0, W1, W2, W3, W4, W6, W9, sep='\n')
    #print([abs(n) for n in [W0, W1, W2, W3, W4, W6, W9]])
    #print(cmath.exp(complex(0, -2*np.pi)))
    W = np.array([[W0, W0, W0, W0], [W0, W1, W2, W3], [W0, W2, W4, W6], [W0, W3, W6, W9]])
    X = np.array(a)
    F = np.dot(W, X)
    print(F)
    print([abs(f) for f in F])

def fft_np():
    a = [11.0, 22.0, 33.0, 44.0]
    b = fft(np.array(a))
    print(b)
    X = [abs(f) for f in b]
    print(X)

def fft(x):
    N = len(x)
    if N <= 1: return x
    even = fft(x[0::2])
    odd =  fft(x[1::2])
    T= [exp(-2j*pi*k/N)*odd[k] for k in range(N//2)]
    return [even[k] + T[k] for k in range(N//2)] + \
           [even[k] - T[k] for k in range(N//2)]

def test():
    print(' '.join("%5.3f" % abs(f) for f in fft([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])))

def test2():
    x=[]
    for i in range(64*64):
        x.append(i % 256) 

    #W = [abs(f) for f in fft(x)]
    W = [abs(f) for f in fft(np.array(x))]
    with open("fft.txt", 'wt') as f:
        for y in range(64):
            for x in range(64):
                f.write("%14.4f, " % W[y*64+x])
            f.write('\n')

def test3():
    w, h = 6, 4
    a = [i for i in range(w*h)]
    f = np.array(a).reshape(h, w)
    F = np.fft.fft2(f)
    F_abs = [abs(a) for a in F]
    with open("fft2d.txt", 'wt') as f:
        for y in range(h):
            for x in range(w):
                f.write("%14.4f, " % F_abs[y][x])
            f.write('\n')
    print('test3 finished')

test3()

print('\ndone')
