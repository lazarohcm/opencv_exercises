import cv2
import numpy as np

cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
img = cv2.imread('./images/reese.jpg')

grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Make the grey scale image have three channels
grey_3_channel = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)

numpy_horizontal = np.hstack((img, grey_3_channel))

numpy_horizontal_concat = np.concatenate((img, grey_3_channel), axis=1)

cv2.imshow('Numpy Horizontal Concat', numpy_horizontal_concat)

cv2.waitKey(0)

cv2.destroyAllWindows()
