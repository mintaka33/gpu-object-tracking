import numpy as np
import cv2

x, y, w, h = (6, 599, 517, 421)
frame = np.fromfile('test.yuv', dtype=np.uint8, count=1920*1080).reshape(1080, 1920)
roi = frame[y:y+h, x:x+w]
roi.tofile('roi-ref.yuv')
cv2.imwrite('roi.bmp', roi)

print('done')