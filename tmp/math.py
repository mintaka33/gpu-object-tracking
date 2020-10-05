import cv2
import numpy as np
from numpy.fft import fft
import cmath
from cmath import exp, pi
import mosse

def dump2txt(filename, a):
    h, w = a.shape
    if type(a[0][0]) is np.complex128:
        cmpl = True
        filename += '.py.%dx%d.txt'%(w*2, h)
    else:
        cmpl = False
        filename += '.py.%dx%d.txt'%(w, h)
    outstr = []
    for y in range(h):
        line = ''
        for x in range(w):
            if cmpl == True:
                line += '%14.6f, %14.6f, '%(a[y][x].real, a[y][x].imag)
            else:
                line += '%14.6f, '%a[y][x]
        outstr.append(line+'\n')

    with open(filename, 'wt') as f:
        for line in outstr:
            f.write(line)

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

def cos_window(sz):
    cos_window = np.hanning(int(sz[1]))[:, np.newaxis].dot(np.hanning(int(sz[0]))[np.newaxis, :])
    return cos_window

def preprocessing(img, cos_window, eps=1e-5):
    img=np.log(img+1)
    img=(img-np.mean(img))/(np.std(img)+eps)
    return cos_window*img

def test_preproc():
    w, h = 20, 10
    f = np.arange((w*h)).reshape((h, w)).astype(np.uint8)
    dump2txt('dump.f.txt', f)
    #dst = np.zeros((h, w)).astype(np.float)
    cos = cos_window((w, h))
    dump2txt('dump.cos.txt', cos)
    dst = preprocessing(f, cos)
    dump2txt('dump.dst.txt', dst)

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

def test4():
    img = cv2.imread('test.bmp')
    h, w = img.shape[:2]
    C = 0.1
    for i in range(8):
        ang = np.random.uniform(-C, C)
        c, s = np.cos(ang), np.sin(ang)
        W = np.array([[c + np.random.uniform(-C, C), -s + np.random.uniform(-C, C), 0],
                      [s + np.random.uniform(-C, C), c + np.random.uniform(-C, C), 0]])
        center_warp = np.array([[w / 2], [h / 2]])
        tmp = np.sum(W[:, :2], axis=1).reshape((2, 1))
        W[:, 2:] = center_warp - center_warp * tmp
        warped = cv2.warpAffine(img, W, (w, h), cv2.BORDER_REFLECT)
        cv2.imwrite('out.%02d.bmp'%i, warped)
    print('test4 finished')

def test_mosse():
    w, h = 640, 360
    r = [387, 198, 30, 62]
    print('frame1 rect:', r)
    y1 = np.zeros((h, w))
    y2 = np.zeros((h, w))
    with open('tmp1.yuv', 'rb') as f:
        yuv = np.fromfile(f, dtype='uint8')
        y1 = yuv[0:w*h].reshape((h, w))
    y1d = cv2.rectangle(y1, (r[0], r[1]), (r[0]+r[2], r[1]+r[3]), (255, 0, 0))
    cv2.imwrite('tmp1.py.out.rect.bmp', y1d)

    mosse.track_init(r, y1)

    dump2txt('dump.py.cos', mosse.cos)
    dump2txt('dump.py.gauss', mosse.gauss)
    dump2txt('dump.py.G', mosse.G)

    #dump2txt('dump.py.Ai.real', mosse.Ai.real)
    #dump2txt('dump.py.Ai.imag', mosse.Ai.imag)
    #dump2txt('dump.py.Bi.real', mosse.Bi.real)
    #dump2txt('dump.py.Bi.imag', mosse.Bi.imag)

    rw, rh = r[2], r[3]
    tmpAi = np.zeros((rh, 2*rw))
    tmpBi = np.zeros((rh, 2*rw))
    tmpAi[:, 0::2] = mosse.Ai.real
    tmpAi[:, 1::2] = mosse.Ai.imag
    tmpBi[:, 0::2] = mosse.Bi.real
    tmpBi[:, 1::2] = mosse.Bi.imag
    dump2txt('dump.py.Ai', tmpAi)
    dump2txt('dump.py.Bi', tmpBi)

    with open('tmp2.yuv', 'rb') as f:
        yuv = np.fromfile(f, dtype='uint8')
        y2 = yuv[0:w*h].reshape((h, w))

    bb = mosse.track_update(y2)
    print('frame2 rect:', bb)
    y2d = cv2.rectangle(y2, (bb[0], bb[1]), (bb[0]+bb[2], bb[1]+bb[3]), (255, 0, 0))
    cv2.imwrite('tmp2.py.out.rect.bmp', y2d)

def test_affine():
    w, h = 30, 62
    img = np.arange(w*h).reshape((h, w)).astype(np.float64)
    out0 = mosse.rand_warp2(img, 0)
    print('finish')

def cos_window(sz):
    #cos_window = np.hanning(int(sz[1]))[:, np.newaxis].dot(np.hanning(int(sz[0]))[np.newaxis, :])
    #cos_window = np.sqrt(cos_window)
    cos_window = cv2.createHanningWindow(sz, cv2.CV_64F)
    return cos_window

def gaussian2d_labels(w, h, sigma):
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    center_x, center_y = w / 2, h / 2
    dist = ((xs - center_x) ** 2 + (ys - center_y) ** 2) / (sigma**2)
    labels = np.exp(-0.5*dist)
    return labels

def affine_matrix(a):
    h, w = a.shape[:2]
    T = np.zeros((2, 3))
    coef = 0.2
    ang = (np.random.rand()-0.5)*coef
    c, s = np.cos(ang), np.sin(ang)
    T[:2, :2] = [[c,-s], [s, c]]
    T[:2, :2] += (np.random.rand(2, 2) - 0.5)*coef
    c = (w/2, h/2)
    T[:,2] = c - np.dot(T[:2, :2], c)
    return T

def genRef():
    w, h = 30, 62
    cos = cos_window((w, h))
    dump2txt('cos', cos)

    g = gaussian2d_labels(w, h, 2.0)
    dump2txt('g', g)

    G = np.fft.fft2(g)
    dump2txt('G', G)

w, h = 30, 62
#test_mosse()
#test_affine()

#genRef()

a = np.arange(w*h).reshape((h, w))
print(affine_matrix(a))

print('\ndone')
