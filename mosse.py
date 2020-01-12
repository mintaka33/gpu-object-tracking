import numpy as np
import cv2 as cv

def cos_window(sz):
    """
    width, height = sz
    j = np.arange(0, width)
    i = np.arange(0, height)
    J, I = np.meshgrid(j, i)
    cos_window = np.sin(np.pi * J / width) * np.sin(np.pi * I / height)
    """
    cos_window = np.hanning(int(sz[1]))[:, np.newaxis].dot(np.hanning(int(sz[0]))[np.newaxis, :])
    return cos_window

def gaussian2d_labels(w, h, sigma):
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    center_x, center_y = w / 2, h / 2
    dist = ((xs - center_x) ** 2 + (ys - center_y) ** 2) / (sigma**2)
    labels = np.exp(-0.5*dist)
    return labels

def preprocessing(img, cos_window, eps=1e-5):
    img=np.log(img+1)
    img=(img-np.mean(img))/(np.std(img)+eps)
    return cos_window*img

def rand_warp(img):
    h, w = img.shape[:2]
    C = .1
    ang = np.random.uniform(-C, C)
    c, s = np.cos(ang), np.sin(ang)
    W = np.array([[c + np.random.uniform(-C, C), -s + np.random.uniform(-C, C), 0],
                  [s + np.random.uniform(-C, C), c + np.random.uniform(-C, C), 0]])
    center_warp = np.array([[w / 2], [h / 2]])
    tmp = np.sum(W[:, :2], axis=1).reshape((2, 1))
    W[:, 2:] = center_warp - center_warp * tmp
    warped = cv.warpAffine(img, W, (w, h), cv.BORDER_REFLECT)
    return warped

pos = [577, 124, 913, 535]
x, y, w, h = pos[0], pos[1], pos[2]-pos[0], pos[3]-pos[1]
center = (x+w/2, y+h/2)
w, h = int(round(w)),int(round(h))
cos = cos_window((w,h))
sigma = 2.0

gauss = gaussian2d_labels(w, h, sigma)
G = np.fft.fft2(gauss)
Ai=np.zeros_like(G)
Bi=np.zeros_like(G)

# Load an color image in grayscale
first = cv.imread('1.bmp', 0)
img1 = first.astype(np.float32)/255
fi = cv.getRectSubPix(img1, (w, h), center)
cv.rectangle(first, (x, y), (x+w, y+h), (255, 0, 0))
cv.imshow('image1', first)

for _ in range(8):
    fi = rand_warp(fi)
    Fi=np.fft.fft2(preprocessing(fi, cos))
    Ai += G * np.conj(Fi)
    Bi += Fi * np.conj(Fi)

second = cv.imread('2.bmp', 0)
img2 = second.astype(np.float32)/255
Hi = Ai/Bi
fi = cv.getRectSubPix(img2, (w, h), center)
fi = preprocessing(fi, cos)
Gi = Hi * np.fft.fft2(fi)
gi = np.real(np.fft.ifft2(Gi))
curr = np.unravel_index(np.argmax(gi, axis=None), gi.shape)
dy, dx = curr[0]-(h/2), curr[1]-(w/2)
dy, dx = int(round(dy)), int(round(dx))
x, y, w, h = x+dx, y+dy, w, h

cv.rectangle(second, (x, y), (x+w, y+h), (255, 0, 0))
cv.imshow('image2', second)
cv.waitKey(0)
cv.destroyAllWindows()

