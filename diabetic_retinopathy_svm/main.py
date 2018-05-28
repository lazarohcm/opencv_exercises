import numpy as np
import time
import sys
import os
import gc

from helpers.class_model import ClassModel
from svm.svm import SVM


sys.settrace
np.set_printoptions(threshold=np.nan)
# Classes
# 0 - No DR
# 1 - Mild
# 2 - Moderate
# 3 - Severe
# 4 - Proliferative DR
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
LABELS = '/trainLabels.csv'
IMAGES_PATH = '/home/lazarohcm/Documents/diabetic_retinopathy/train'

# HOG Parameters
CELL_SIZE = (128, 128)  # Loading AR training set
BLOCK_SIZE = (4, 4)
NBINS = 9
SAMPLES_SIZE = 200
RESIZE_TO = (1024, 768)

t_start = time.time()

no_dr = ClassModel(0, CURRENT_PATH + LABELS, IMAGES_PATH, SAMPLES_SIZE)
p_dr = ClassModel(4, CURRENT_PATH + LABELS, IMAGES_PATH, SAMPLES_SIZE)
svm = SVM([no_dr, p_dr], CELL_SIZE, BLOCK_SIZE, NBINS, RESIZE_TO)

# Training
svm.train()
print("Time Taken on training:", np.round(time.time() - t_start, 2))
gc.collect()

# Testing
t_start = time.time()
svm.test()
print("Time Taken on training:", np.round(time.time() - t_start, 2))
gc.collect()

svm.save_test('classes_0_4.csv')
