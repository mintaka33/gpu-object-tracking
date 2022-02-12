import os
import numpy as np
import cv2
import glob

srcw, srch = 1920, 1080
x, y, w, h = 6, 599, 517, 421
app_name = 'gpu_math.exe'
app_dir = 'D:\\Code\\gpu_tracking\\gpu-object-tracking\\build\\bin'
yuv_file = '%s\\test.yuv'%app_dir
roi_file = '%s\\dump.gpu-roi.0000.517x421.yuv'%app_dir
aff_file = '%s\\dump.gpu-affine.0000.517x421.yuv'%app_dir
proc_file = '%s\\dump.0000.gpu-preproc.1034x421.txt'%app_dir
cos2d_file = '%s\\dump.0000.gpu-cos2d.517x421.txt'%app_dir
R_file = '%s\\dump.0000.gpu-r.1034x421.txt'%app_dir

def execute(cmd):
    print('#'*8, cmd)
    os.system(cmd)

def dump_result(data, tag):
    filename = '%s\\dump_%s_%dx%d.txt' % (app_dir, tag, data.shape[1], data.shape[0])
    np.savetxt(filename, data, fmt='%+.18e', delimiter=', ')

def verify_affine():
    # gpu result
    cmd = 'cd %s && %s' % (app_dir, app_name)
    execute(cmd)
    frame = np.fromfile(roi_file, dtype=np.uint8, count=w*h).reshape((h, w))
    cv2.imwrite('%s\\roi.bmp' % app_dir, frame)
    frame = np.fromfile(aff_file, dtype=np.uint8, count=w*h).reshape((h, w))
    cv2.imwrite('%s\\aff.bmp' % app_dir, frame)
    # ref result
    yuv = np.fromfile(yuv_file, dtype=np.uint8, count=srcw*srch).reshape((srch, srcw))
    a = yuv[y:y+h, x:x+w]
    T = np.array([[1.021916, -0.021326, -1.176091], [0.039830, 0.923501, 5.806976]])
    b = cv2.warpAffine(a, T, (w, h), flags = cv2.INTER_LINEAR, borderMode = cv2.BORDER_REFLECT)
    cv2.imwrite('%s\\ref.bmp'%app_dir, b)

def verify_fft():
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
        app_cmd = '%s %d %d' % (app_name, w, h)
        cmd = 'cd %s && %s' % (app_dir, app_cmd)
        execute(cmd)
        filename = 'dump.0000.gpu-fft.%dx%d.txt' % (w*2, h)
        result = np.genfromtxt('%s\\%s'%(app_dir, filename), dtype=np.float64, delimiter=",")
        result = result[::, :-1:]
        # r, i = result[:, 0::2], result[:, 1::2]
        return result

    w, h = 53, 31
    gpu = gpu_fft(w, h)
    dump_result(gpu, 'gpu')
    ref = ref_fft(w, h)
    dump_result(ref, 'ref')

    # print('INFO: [%dx%d] sum of delta = %f, max = %f' % (w, h, np.sum(np.abs(ref - gpu)), np.max(np.abs(ref - gpu))))

def verify_preproc():
    # gpu result
    cmd = 'cd %s && %s' % (app_dir, app_name)
    execute(cmd)

    # reference result
    yuv = np.fromfile(yuv_file, dtype=np.uint8, count=srcw*srch).reshape((srch, srcw))
    crop = yuv[y:y+h, x:x+w].astype(np.uint8)
    crop.tofile('%s\\ref_crop.yuv'%app_dir)
    norm = np.log(np.float64(crop)+1)
    dump_result(norm, 'ref_norm')

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
# verify_fft()
verify_preproc()
# find_max()

print('done')