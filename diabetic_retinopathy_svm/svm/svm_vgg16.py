import os
# os.environ["CUDA_DEVICE_ORDER"] = "03:00.0"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import sys
import numpy as np
import csv
from helpers.class_model import ClassModel
from typing import List

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image as preprocessor
from keras.applications.vgg16 import preprocess_input


# Usar Keras
# Keras -> deep features
# VGG 16/19 e RESNET

# Testar autosklearn -> Estimador de classificador

class SVM16:
    Classes = List[ClassModel]
    model = None

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
        self.model = VGG16(weights='imagenet', include_top=False)

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

    def load_samples(self, stage, samples, image_list):
        images_path = samples[:, 0]

        total = len(images_path)
        print('Starting to' + stage + 'with vgg16')
        for index, path in enumerate(images_path):
            per = ((index + 1) / float(total)) * 100
            sys.stdout.write("\r %s %d%% " % ('Reading Images... ', per))
            sys.stdout.flush()
            img = preprocessor.load_img(path, target_size=(224, 224))
            img = preprocessor.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)
            image_list.append(img)

    def train(self, mode='auto'):
        self.load_samples('train', self.train_samples, self.train_images)
        total = len(self.train_images)
        print('')

        for index, image in enumerate(self.train_images):
            per = ((index + 1) / float(total)) * 100
            sys.stdout.write("\r %s %d%% " %
                             ('Extrating features with VGG16... ', per))
            sys.stdout.flush()
            vgg16_features = np.float32(self.model.predict(image))
            vgg16_features = np.transpose(vgg16_features)
            if(len(self.train_features) == 0):
                self.train_features = vgg16_features
            else:
                self.train_features = vgg16_features if len(
                    self.train_features) == 0 else np.concatenate((self.train_features, vgg16_features), axis=0)

        print(self.train_features.shape)
        # return
        if(mode == 'auto'):
            self.svm.trainAuto(self.train_features, cv2.ml.ROW_SAMPLE,
                               self.train_samples, 10)
        elif(mode == 'defined'):
            self.svm.train(self.train_features, cv2.ml.ROW_SAMPLE,
                           self.train_samples[:, 1].astype(int), 10)

        self.svm.save('training_result.dat')

    def test(self, mode=None):
        # Cleaning memory
        self.train_images = None
        self.train_features = None
        self.train_samples = None
        print('Testing')
        if(mode == 'vgg16'):
            self.load_vgg16_samples(
                'test', self.test_samples, self.test_images)

        self.load_samples('test', self.test_samples, self.test_images)
        total = len(self.test_images)
        print('')
        for index, image in enumerate(self.test_images):
            per = ((index + 1) / float(total)) * 100
            sys.stdout.write("\r %s %d%% " % ('Computing Hogs... ', per))
            sys.stdout.flush()

            vgg16_features = np.float32(self.model.predict(image))
            vgg16_features = np.transpose(vgg16_features)

            if(len(self.test_features) == 0):
                self.test_features = vgg16_features
            else:
                self.test_features = vgg16_features if len(
                    self.test_features) == 0 else np.concatenate((self.test_features, vgg16_features), axis=0)

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
