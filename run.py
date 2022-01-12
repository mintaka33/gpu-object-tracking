import os
import numpy as np
import cv2

w, h = 517, 421
app_name = 'gpu_math.exe'
app_dir = 'D:\\Code\\gpu_tracking\\gpu-object-tracking\\build\\bin'
roi_file = 'dump.gpu-roi.0000.517x421.yuv'
aff_file = 'dump.gpu-affine.0000.517x421.yuv'

def execute(cmd):
    print('#'*8, cmd)
    os.system(cmd)

def verify_affine():
    cmd = 'cd %s && %s' % (app_dir, app_name)
    execute(cmd)

    frame = np.fromfile('%s\\%s'%(app_dir, roi_file), dtype=np.uint8, count=w*h).reshape((h, w))
    cv2.imwrite('%s\\roi.bmp' % app_dir, frame)
    frame = np.fromfile('%s\\%s'%(app_dir, aff_file), dtype=np.uint8, count=w*h).reshape((h, w))
    cv2.imwrite('%s\\aff.bmp' % app_dir, frame)

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

def verify_preproc():
    cmd = 'cd %s && %s' % (app_dir, app_name)
    execute(cmd)
    aff = np.fromfile('%s\\%s' % (app_dir, aff_file), dtype=np.uint8).reshape((h, w))
    aff2 = np.log(aff.astype(np.float))
    aff3 = np.round(aff2).astype(np.int) # round float to int to match with kernel implementation
    print('affine sum = %f, average = %f' % (np.sum(aff3), np.average(aff3)))
    avg = np.average(aff3)
    aff4 = (aff2 - avg) * (aff2 - avg)
    aff4 = np.round(aff4).astype(np.int)
    print('affine std_sum = %f, std = %f' % (np.sum(aff4), np.sqrt(np.sum(aff4)/(w*h))))
    pass

# verify_affine()
# verify_fft()

verify_preproc()

print('done')