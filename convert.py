import sys
import os
import glob

for f in glob.glob('./*.yuv'): 
	if 'out' in f:
		base = os.path.basename(f)
		(name, ext) = os.path.splitext(base)
		_, w, h, i = name.split('_')
		cmd = 'ffmpeg -y -s %sx%s -pix_fmt gray -i %s out.%s.bmp' % (w, h, f, i)  
		print(cmd) 
		os.system(cmd)

print('done')