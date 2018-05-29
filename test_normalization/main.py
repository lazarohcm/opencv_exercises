import cv2
import glob
import numpy as np

np.set_printoptions(threshold=np.nan)
images = []
images_name = images_name = glob.glob('sample/*')

for index, image in enumerate(images_name):
    print('Reading ' + str(index) + ' of ' + str(len(images_name)))
    img = cv2.imread(image)
    img = cv2.resize(img, (1024, 768), interpolation=cv2.INTER_AREA)

    height, width, depth = img.shape
    norm = img.copy()

    b, g, r = cv2.split(img)
    sum = b+g+r
    norm[:, :, 0] = b/sum*255.0
    norm[:, :, 1] = g/sum*255.0
    norm[:, :, 2] = r/sum*255.0
    # if(index == len(images_name) - 1):
    cv2.imshow(image, norm)

cv2.waitKey(0)
cv2.destroyAllWindows()
