import numpy as np
import cv2

Ai, Bi, G, cos, gauss, f_rect, fip, Fi, H, Gi, Hi = None, None, None, None, None, None, None, None, None, None, None
x, y, w, h, center, gi = None, None, None, None, None, None
sigma = 2.0
interp_factor = 0.125

def cos_window(sz):
    cos_window = np.hanning(int(sz[1]))[:, np.newaxis].dot(np.hanning(int(sz[0]))[np.newaxis, :])
    cos_window = np.sqrt(cos_window)
    #cos_window = cv2.createHanningWindow(sz, cv2.CV_64F)
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
    tmp = cos_window*img
    return tmp

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
    warped = cv2.warpAffine(img, W, (w, h), cv2.BORDER_REFLECT)
    return warped

def rand_warp2(a):
    h, w = a.shape[:2]
    T = np.zeros((2, 3))
    coef = 0.2
    ang = (np.random.rand()-0.5)*coef
    c, s = np.cos(ang), np.sin(ang)
    T[:2, :2] = [[c,-s], [s, c]]
    T[:2, :2] += (np.random.rand(2, 2) - 0.5)*coef
    c = (w/2, h/2)
    T[:,2] = c - np.dot(T[:2, :2], c)
    #T = np.array([[1.027946, 0.003986, -0.542760], [-0.142644, 1.008884, 1.864269]])
    return cv2.warpAffine(a, T, (w, h), borderMode = cv2.BORDER_REFLECT)

def track_init(pos, frame):
    global Ai, Bi, G, cos, gauss, x, y, w, h, center, f_rect, fip, Fi, H
    x, y, w, h = pos[0], pos[1], pos[2], pos[3]
    center = (x+w/2, y+h/2)
    cos = cos_window((w, h))

    gauss = gaussian2d_labels(w, h, sigma)
    G = np.fft.fft2(gauss)
    Ai = np.zeros_like(G)
    Bi = np.zeros_like(G)

    # Load an color image in grayscale
    img1 = frame.astype(np.float32)/255
    #f_rect = cv2.getRectSubPix(img1, (w, h), center)
    f_rect = img1[y:y+h, x:x+w]

    for _ in range(8):
        fi = rand_warp2(f_rect)
        fip = preprocessing(fi, cos)
        Fi = np.fft.fft2(fip)
        Ai += G * np.conj(Fi)
        Bi += Fi * np.conj(Fi)
    H = Ai/Bi

def track_update(frame):
    global Ai, Bi, G, cos, x, y, w, h, center, f_rect, fi, fip, Fi, Gi, Hi, gi
    img2 = frame.astype(np.float32)/255
    Hi = Ai/Bi
    #fi = cv2.getRectSubPix(img2, (w, h), center)
    fi = img2[y:y+h, x:x+w]
    fip = preprocessing(fi, cos)
    Fi = np.fft.fft2(fip)
    Gi = Hi * Fi
    gi = np.real(np.fft.ifft2(Gi))
    curr = np.unravel_index(np.argmax(gi, axis=None), gi.shape)
    dy, dx = int(round(curr[0]-(h/2))), int(round(curr[1]-(w/2)))
    bb = [x+dx, y+dy, w, h]
    xc, yc = center
    xc += dx
    yc += dy
    x, y = (x+dx, y+dy)
    center = (xc, yc)
    f_rect = cv2.getRectSubPix(img2, (w, h), center)
    Fi = np.fft.fft2(preprocessing(f_rect, cos))
    Ai = interp_factor * (G * np.conj(Fi)) + (1 - interp_factor) * Ai
    Bi = interp_factor * (Fi * np.conj(Fi)) + (1 - interp_factor) * Bi
    return bb

def main():
    cap = cv2.VideoCapture('test.265')
    if not cap.isOpened():
        print("ERROR: cannot open video file!")
        exit()

    init_bb = None
    while True:
        ret, frame = cap.read()
        if ret ==  False:
            break
        frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        if init_bb is None:
            init_bb = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
            track_init(init_bb, frame_gray)
            x, y, w, h = init_bb[0], init_bb[1], init_bb[2], init_bb[3]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            print(init_bb)
        else:
            out_bb = track_update(frame_gray)
            x, y, w, h = out_bb[0], out_bb[1], out_bb[2], out_bb[3]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            print(out_bb)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    print('done')

