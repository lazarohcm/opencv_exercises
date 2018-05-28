import numpy as np
import csv
import cv2
import sys
import time
import random


class ClassModel:
    labels_path = ''
    images_path = ''

    def __init__(self, label='', labels_path='', images_path='', samples=[]):
        self.samples = []
        self.train_samples = []
        self.test_samples = []
        self.train_samples_image = []
        self.test_samples_image = []

        self.label = label
        self.labels_path = labels_path
        self.images_path = images_path

        self.load_labels()
        self.separate_samples()

    def load_labels(self):
        with open(self.labels_path) as images_label:
            reader = csv.DictReader(images_label, delimiter=',')
            i = 0
            for row in reader:
                # print(index)
                if(int(row['level']) == self.label):
                    i += 1
                    # print (i)
                    self.samples.append([
                        row['image'] + '.' + 'jpeg',
                        int(row['level'])
                    ])
        print('Total of class ' + str(self.label) +
              ' -> ' + str(len(self.samples)))

    def separate_samples(self):
        self.train_samples = random.sample(self.samples, 500)
        self.test_samples = random.sample(
            self.diff(self.samples, self.train_samples), 500)

    def diff(self, first, second):
        list_diff = []
        for item in first:
            if(item not in second):
                list_diff.append(item)
        return list_diff

    def load_train(self):
        total = len(self.train_samples)
        for index, label in enumerate(self.train_samples):
            image_path = self.images_path + '/' + label[0]
            per = ((index + 1) / float(total)) * 100
            sys.stdout.write("\r %s %d%% " % ('Reading Images... ', per))
            sys.stdout.flush()
            self.train_samples_image.append(cv2.imread(image_path))
