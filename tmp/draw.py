import os, sys
import cv2
import numpy as np

roifile = 'C:\\data\\code\\mintaka33\\visual-object-tracking\\build\\Debug\\out.txt'
pw, ph = 640, 360
x, y, w, h = [270, 160, 53, 33]

rois = []
with open(roifile, 'rt') as f:
    rois = f.readlines()

for i, l in enumerate(rois) :
	a, b, c, d = l.split('\n')[0].split('INFO: ')[1].split(', ')
	dx = int(c.split(' = ')[1])
	dy = int(d.split(' = ')[1])
	x, y = x + dx, y + dy
	infile = 'input2\\tmp.%03d.yuv'%(i+2)
	outfile = 'output\\tmp.%03d.bmp'%(i+2)
	with open(infile, 'rb') as f:
		yp = np.fromfile(f, dtype='uint8', count=pw*ph).reshape((ph, pw))
	cv2.rectangle(yp, (x, y), (x+w, y+h), (255, 0, 0))
	cv2.imwrite(outfile, yp)

print('done')