import cv2
import numpy as np
import glob
import time
import sys
import csv
from helpers.class_model import ClassModel
import os

# Classes
# 0 - No DR
# 1 - Mild
# 2 - Moderate
# 3 - Severe
# 4 - Proliferative DR
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
LABELS = '/trainLabels.csv'
IMAGES_PATH = '/home/lazarohcm/Documents/diabetic_retinopathy/train'

no_dr = ClassModel(0, CURRENT_PATH + LABELS, IMAGES_PATH)
p_dr = ClassModel(1, CURRENT_PATH + LABELS, IMAGES_PATH)

sys.settrace
np.set_printoptions(threshold=np.nan)

print('Starting train')
no_dr.load_train()

# train_images = Images(CURRENT_PATH + '/kaggle_small_train/')
# train_images.load_labels(CURRENT_PATH + '/trainLabels.csv', ',')

# print(train_images.classes)
# train_images.open_images()

# HOG Parameters
CELL_SIZE = (128, 128)  # Loading AR training set
BLOCK_SIZE = (4, 4)
NBINS = 9

results = []
labels = []
with open("kaggle_small_train_labels.csv") as trains_labels:
    # change contents to floats
    reader = csv.DictReader(trains_labels, delimiter=',')
    for row in reader:  # each row is a list
        results.append(row)
        labels.append(int(row['level']))
labels = np.array(labels)


def defineHOG(image, cell_size, block_size, nbins):
    win_size = (image.shape[1] // cell_size[1] * cell_size[1],
                image.shape[0] // cell_size[0] * cell_size[0])
    block = (block_size[1] * cell_size[1], block_size[0] * cell_size[0])
    stride = (cell_size[1], cell_size[0])
    cell_size = (cell_size[1], cell_size[0])
    return cv2.HOGDescriptor(win_size, block, stride, cell_size, nbins)


t_start = time.time()


# Read training images
images_name = glob.glob('kaggle_small_train/*')
test_images_name = glob.glob('kaggle_small_test/*')

images = []

# Loading images for training
for index, image in enumerate(images_name):
    print('Reading ' + str(index) + ' of ' + str(len(images_name)))
    img = cv2.imread(image)
    img = cv2.resize(img, (1024, 768), interpolation=cv2.INTER_AREA)
    images.append(img)
    # if(index == 5):
    # break
test_images = []

images = np.asarray(images)

features = []

# Getting features
for image in images:
    n_cells = (image.shape[0] // CELL_SIZE[0], image.shape[1] // CELL_SIZE[1])
    hog = np.float32(defineHOG(image, CELL_SIZE, BLOCK_SIZE,
                               NBINS).compute(image))
    hog = np.transpose(hog)
    if(len(features) == 0):
        features = hog
    else:
        features = np.concatenate((features, hog))

features = np.array(features)

print(features.shape)

svm = cv2.ml.SVM_create()
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setType(cv2.ml.SVM_C_SVC)
svm.trainAuto(features, cv2.ml.ROW_SAMPLE, labels, 10)
svm.save('svm_data.dat')

print("Time Taken on training:", np.round(time.time() - t_start, 2))


for index, image in enumerate(test_images_name):
    print('Reading test ' + str(index) + ' of ' + str(len(test_images_name)))
    img = cv2.imread(image)
    img = cv2.resize(img, (1024, 768), interpolation=cv2.INTER_AREA)
    test_images.append(img)
test_images = np.asarray(images)
print(len(test_images))
test_features = []

# Getting features
for image in test_images:
    n_cells = (image.shape[0] // CELL_SIZE[0], image.shape[1] // CELL_SIZE[1])
    hog = np.float32(defineHOG(image, CELL_SIZE, BLOCK_SIZE,
                               NBINS).compute(image))
    hog = np.transpose(hog)
    if(len(test_features) == 0):
        test_features = hog
    else:
        test_features = np.concatenate((test_features, hog))
test_features = np.array(test_features)

res = svm.predict(test_features)
print(res)
