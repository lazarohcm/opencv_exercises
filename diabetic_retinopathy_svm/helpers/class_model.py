import numpy as np
import csv
import cv2
import sys
import time
import random


class ClassModel:
    labels_path = ''
    images_path = ''
    # samples_size = 500

    def __init__(self, label='', labels_path='', images_path='', samples_size=500):
        self.samples = []
        self.train_samples = []
        self.test_samples = []
        self.samples_size = samples_size

        self.label = label
        self.labels_path = labels_path
        self.images_path = images_path

        self.load_labels()
        self.separate_samples()

    def load_labels(self):
        with open(self.labels_path) as images_label:
            reader = csv.DictReader(images_label, delimiter=',')
            for row in reader:
                # print(index)
                if(int(row['level']) == self.label):
                    image = self.images_path + '/' + \
                        row['image'] + '.' + 'jpeg'
                    self.samples.append([image, int(row['level'])])
        print('Total of class ' + str(self.label) +
              ' -> ' + str(len(self.samples)))

    def separate_samples(self):
        self.train_samples = random.sample(self.samples, self.samples_size)
        self.test_samples = random.sample(
            self.diff(self.samples, self.train_samples), self.samples_size)

    def diff(self, first, second):
        list_diff = []
        for item in first:
            if(item not in second):
                list_diff.append(item)
        return list_diff
