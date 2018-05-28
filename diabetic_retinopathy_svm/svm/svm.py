import cv2
import sys
import numpy as np
import csv
from helpers.class_model import ClassModel
from typing import List


class SVM:
    Classes = List[ClassModel]

    def __init__(self, classes_model: Classes, cell_size, block_size, n_bins, resize_to=None):
        self.cell_size = cell_size
        self.block_size = block_size
        self.n_bins = n_bins
        self.train_images = []
        self.test_images = []
        self.train_samples = np.array([])
        self.test_samples = np.array([])
        self.classes_model = classes_model
        self.svm = None
        self.setup()
        self.train_features = np.array([])
        self.test_features = np.array([])
        self.resize_to = resize_to
        self.test_results = []

    def define_hog(self, image, cell_size, block_size, n_bins):
        win_size = (image.shape[1] // cell_size[1] * cell_size[1],
                    image.shape[0] // cell_size[0] * cell_size[0])
        block = (block_size[1] * cell_size[1], block_size[0] * cell_size[0])
        stride = (cell_size[1], cell_size[0])
        cell_size = (cell_size[1], cell_size[0])
        return cv2.HOGDescriptor(win_size, block, stride, cell_size, n_bins)

    def compute_hog(self, image):
        hog = np.float32(self.define_hog(image, self.cell_size, self.block_size,
                                         self.n_bins).compute(image))
        hog = np.transpose(hog)
        return hog

    def setup(self):
        # Train Samples
        samples = np.array([]).reshape(0, 2)
        for class_model in self.classes_model:
            samples = np.concatenate(
                (samples, class_model.train_samples), axis=0)
        # shuffling training samples
        self.train_samples = np.array(np.random.permutation(samples))

        # Test Samples
        samples = np.array([]).reshape(0, 2)
        for class_model in self.classes_model:
            samples = np.concatenate(
                (samples, class_model.test_samples), axis=0)
        # shuffling training samples
        self.test_samples = np.array(np.random.permutation(samples))

        # SVM
        self.svm = cv2.ml.SVM_create()
        self.svm.setKernel(cv2.ml.SVM_LINEAR)
        self.svm.setType(cv2.ml.SVM_C_SVC)

    def load_samples(self, stage, samples, images_list):
        images_path = samples[:, 0]

        total = len(images_path)
        print('Starting ' + stage)
        for index, path in enumerate(images_path):
            per = ((index + 1) / float(total)) * 100
            sys.stdout.write("\r %s %d%% " % ('Reading Images... ', per))
            sys.stdout.flush()
            if(self.resize_to is None):
                images_list.append(cv2.imread(path))
            else:
                img = cv2.imread(path)
                images_list.append(cv2.resize(
                    img, (self.resize_to), interpolation=cv2.INTER_AREA))

    def train(self):
        self.load_samples('train', self.train_samples, self.train_images)
        total = len(self.train_images)
        print('')
        for index, image in enumerate(self.train_images):
            per = ((index + 1) / float(total)) * 100
            sys.stdout.write("\r %s %d%% " % ('Computing Hogs... ', per))
            sys.stdout.flush()
            # hog_features = self.compute_hog(image)
            if(len(self.train_features) == 0):
                self.train_features = self.compute_hog(image)
            else:
                self.train_features = self.compute_hog(image) if len(
                    self.train_features) == 0 else np.concatenate((self.train_features, self.compute_hog(image)), axis=0)

        self.svm.trainAuto(self.train_features, cv2.ml.ROW_SAMPLE,
                           self.train_samples[:, 1].astype(int), 10)
        self.svm.save('training_result.dat')

    def test(self):
        # Cleaning memory
        self.train_images = None
        self.train_features = None
        self.train_samples = None
        print('Testing')
        self.load_samples('test', self.test_samples, self.test_images)
        total = len(self.test_images)
        print('')
        for index, image in enumerate(self.test_images):
            per = ((index + 1) / float(total)) * 100
            sys.stdout.write("\r %s %d%% " % ('Computing Hogs... ', per))
            sys.stdout.flush()
            if(len(self.test_features) == 0):
                self.test_features = self.compute_hog(image)
            else:
                self.test_features = self.compute_hog(image) if len(
                    self.test_features) == 0 else np.concatenate((self.test_features, self.compute_hog(image)), axis=0)

        self.test_results = self.svm.predict(self.test_features)[1].ravel()

    def save_test(self, file_name):
        with open(file_name, 'w') as csvfile:
            fieldnames = ['prediction', 'actual_class']
            writer = csv.DictWriter(
                csvfile, delimiter=",", fieldnames=fieldnames)
            writer.writeheader()
            for index, result in enumerate(self.test_results):
                writer.writerow(
                    {'prediction': int(result), 'actual_class': self.test_samples[:, 1][index]})
