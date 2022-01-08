import os
import numpy as np
import cv2

def execute(cmd):
    print('#'*8, cmd)
    os.system(cmd)

cmd = 'cd build/bin && gpu_math.exe'
execute(cmd)

w, h = 517, 421
frame = np.fromfile('aff.yuv', dtype=np.uint8, count=w*h).reshape((h, w))
cv2.imwrite('aff.bmp', frame)

print('done')