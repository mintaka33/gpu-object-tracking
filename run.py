import os
import numpy as np
import cv2
import glob

w, h = 517, 421
app_name = 'gpu_math.exe'
app_dir = 'D:\\Code\\gpu_tracking\\gpu-object-tracking\\build\\bin'
roi_file = '%s\\dump.gpu-roi.0001.517x421.yuv'%app_dir
aff_file = '%s\\dump.gpu-affine.0001.517x421.yuv'%app_dir
proc_file = '%s\\dump.0000.gpu-preproc.1034x421.txt'%app_dir
cos2d_file = '%s\\dump.0000.gpu-cos2d.517x421.txt'%app_dir
R_file = '%s\\dump.0000.gpu-r.1034x421.txt'%app_dir

def execute(cmd):
    print('#'*8, cmd)
    os.system(cmd)

def verify_affine():
    cmd = 'cd %s && %s' % (app_dir, app_name)
    # execute(cmd)

    frame = np.fromfile(roi_file, dtype=np.uint8, count=w*h).reshape((h, w))
    cv2.imwrite('%s\\roi.bmp' % app_dir, frame)
    frame = np.fromfile(aff_file, dtype=np.uint8, count=w*h).reshape((h, w))
    cv2.imwrite('%s\\aff.bmp' % app_dir, frame)

def verify_fft():
    def dump_result(data, tag):
        filename = '%s\\dump_%s_%dx%d.txt' % (app_dir, tag, data.shape[1], data.shape[0])
        np.savetxt(filename, data, fmt='%-15f', delimiter=', ')
    def gaussian2(w, h, sigma=2.0):
        xs, ys = np.meshgrid(np.arange(w), np.arange(h))
        center_x, center_y = w / 2, h / 2
        dist = ((xs - center_x) ** 2 + (ys - center_y) ** 2) / (sigma**2)
        g = np.exp(-0.5*dist).astype(np.float64)
        return g
    def get_input(w, h):
        filename = 'dump.0000.input.%dx%d.txt' % (w, h)
        data = np.genfromtxt('%s\\%s'%(app_dir, filename), dtype=np.float64, delimiter=",")
        data = data[::, :-1:]
        return data.reshape((h, w))
    def ref_fft(w, h):
        g = get_input(w, h) # gaussian2(w, h)
        dump_result(g, 'input')
        # G = cv2.dft(g, flags = cv2.DFT_COMPLEX_OUTPUT)
        G = np.fft.fft2(g)
        result = np.zeros((h, w*2), dtype=np.float64)
        result[:, 0::2] = G.real
        result[:, 1::2] = G.imag
        return result
    def gpu_fft(w, h):
        cmd = 'cd %s && %s' % (app_dir, app_name)
        execute(cmd)
        filename = 'dump.0000.gpu-fft.%dx%d.txt' % (w*2, h)
        result = np.genfromtxt('%s\\%s'%(app_dir, filename), dtype=np.float64, delimiter=",")
        result = result[::, :-1:]
        # r, i = result[:, 0::2], result[:, 1::2]
        return result

    w, h = 16, 16
    gpu = gpu_fft(w, h)
    dump_result(gpu, 'gpu')
    ref = ref_fft(w, h)
    dump_result(ref, 'ref')

    # print('INFO: [%dx%d] sum of delta = %f, max = %f' % (w, h, np.sum(np.abs(ref - gpu)), np.max(np.abs(ref - gpu))))

def verify_preproc():
    # gpu result
    cmd = 'cd %s && %s' % (app_dir, app_name)
    execute(cmd)
    gpu_cos2d = np.genfromtxt(cos2d_file, dtype=float, delimiter=',')
    gpu_cos2d = gpu_cos2d[:, :-1]
    gpu_proc = np.genfromtxt(proc_file, dtype=float, delimiter=',')
    gpu_proc = gpu_proc[:, :-1]
    gpu_proc = gpu_proc[:, 0::2] # skip imaginary 
    # reference result
    aff = np.fromfile(aff_file, dtype=np.uint8).reshape((h, w))
    cos2d = cv2.createHanningWindow((w, h), cv2.CV_32F)
    np.savetxt('%s\\ref.cos2d.txt'%app_dir, cos2d, fmt='%-14.6f', delimiter=', ')
    print('cos-diff = %f' % np.sum(np.abs(cos2d -gpu_cos2d)))

    ref = np.log(np.float32(aff) + 1.0)
    ref = (ref - ref.mean()) / (ref.std() + 1e-5)
    ref = ref * cos2d
    np.savetxt('%s\\ref.proc.txt'%app_dir, ref, fmt='%-14.6f', delimiter=', ')
    print('proc-diff = %f' % np.sum(np.abs(ref -gpu_proc)))

def yuv_to_image():
    for yuvfile in glob.glob('%s\\dump.*.yuv'%app_dir):
        imgfile = '%s.bmp' % yuvfile
        data = np.fromfile(yuvfile, dtype=np.uint8, count=w*h).reshape((h, w))
        cv2.imwrite(imgfile, data)

def find_max():
    r = np.genfromtxt(R_file, dtype=float, delimiter=',')
    r = r[:, 0::2]
    idx = np.unravel_index(r.argmax(), r.shape)
    print(idx)

# yuv_to_image()
# verify_affine()
verify_fft()
# verify_preproc()

# find_max()

print('done')