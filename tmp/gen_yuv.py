import sys
import os
import glob

for i in range(250):
	bmpfile = 'input\\tmp.%03d.bmp'%(i+1)
	yuvfile = 'input2\\tmp.%03d.yuv'%(i+1)
	cmd = 'ffmpeg -y -i %s -pix_fmt yuv420p %s' % (bmpfile, yuvfile)
	#print(cmd) 
	os.system(cmd)

print('done')