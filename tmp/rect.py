import cv2

#cap = cv2.VideoCapture('build/Debug/tmp1.bmp')
frame = cv2.imread('build/Debug/tmp1.bmp')
roi = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
print(roi)

print('done')
