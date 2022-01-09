import os
import numpy as np
import cv2

w, h = 517, 421
app_dir = 'D:\\Code\\gpu_tracking\\gpu-object-tracking\\build\\bin'
app_name = 'gpu_math.exe'

def execute(cmd):
    print('#'*8, cmd)
    os.system(cmd)

def verify_crop():
    cmd = 'cd %s && %s' % (app_dir, app_name)
    execute(cmd)
    frame = np.fromfile('aff.yuv', dtype=np.uint8, count=w*h).reshape((h, w))
    cv2.imwrite('aff.bmp', frame)

def verify_fft():
    def dump_result(data, tag):
        filename = '%s\\dump_%s_%dx%d.txt' % (app_dir, tag, data.shape[1], data.shape[0])
        np.savetxt(filename, data, fmt='%-14f', delimiter=', ')
    def gaussian2(w, h, sigma=2.0):
        xs, ys = np.meshgrid(np.arange(w), np.arange(h))
        center_x, center_y = w / 2, h / 2
        dist = ((xs - center_x) ** 2 + (ys - center_y) ** 2) / (sigma**2)
        g = np.exp(-0.5*dist).astype(np.float64)
        return g
    def numpy_fft(w, h):
        g = gaussian2(w, h)
        G = np.fft.fft2(g)
        result = np.zeros((h, w*2), dtype=np.float64)
        result[:, 0::2] = G.real
        result[:, 1::2] = G.imag
        return result
    def gpu_fft(w, h):
        cmd = 'cd %s && %s' % (app_dir, app_name)
        execute(cmd)
        result = np.genfromtxt('%s\\result.txt'%app_dir, dtype=np.float64, delimiter=",")
        # r, i = result[:, 0::2], result[:, 1::2]
        return result[:, :-1]
    
    ref = numpy_fft(w, h)
    dump_result(ref, 'ref')
    gpu = gpu_fft(w, h)
    dump_result(gpu, 'gpu')
    print('INFO: [%dx%d] sum of delta = %f, max = %f' % (w, h, np.sum(np.abs(ref - gpu)), np.max(np.abs(ref - gpu))))

# verify_crop()
verify_fft()

print('done')