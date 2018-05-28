import cv2
import os
import numpy as np
import csv


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append([filename, img])
    return images


def defineHOG(image, cell_size, block_size, nbins):
    win_size = (image.shape[1] // cell_size[1] * cell_size[1],
                image.shape[0] // cell_size[0] * cell_size[0])
    block = (block_size[1] * cell_size[1], block_size[0] * cell_size[0])
    stride = (cell_size[1], cell_size[0])
    cell_size = (cell_size[1], cell_size[0])
    return cv2.HOGDescriptor(win_size, block, stride, cell_size, nbins)


def computeHOG(image, hog, cell_size, block_size, nbins):
    n_cells = (image.shape[0] // cell_size[0], image.shape[1] // cell_size[1])
    return hog.compute(image)\
        .reshape(n_cells[1] - BLOCK_SIZE[1] + 1,
                 n_cells[0] - BLOCK_SIZE[0] + 1,
                 BLOCK_SIZE[0], BLOCK_SIZE[1], NBINS) \
        .transpose((1, 0, 2, 3, 4))


def getGradients(folder_path, cell_size, block_size, nbins):
    image_set = load_images_from_folder(folder_path)
    gradients = []
    for index, image in enumerate(image_set):
        # Tamanho de janela
        filename = image[0]
        print(filename + ' is ' + str(index) + ' of ' + str(len(image_set)))

        hog = defineHOG(image[1], cell_size, block_size, nbins)
        gradient = computeHOG(image[1], hog, cell_size, block_size, nbins)
        print(gradient.size)
        gradients.append([filename, np.float32(gradient)])
        # if index == 3: break
    return gradients


# HOG Parameters
CELL_SIZE = (256, 256)  # Loading AR training set
# ar_training_images = load_images_from_folder('./training/apparent_retinopathy/')
BLOCK_SIZE = (4, 4)
NBINS = 9
AR_FOLDER = './training/apparent_retinopathy/'
NAR_FOLDER = './training/no_apparent_retinopathy/'

# Loading AR training set
ar_gradients = getGradients(AR_FOLDER, CELL_SIZE, BLOCK_SIZE, NBINS)

# Loading NAR training set
nar_gradients = getGradients(NAR_FOLDER, CELL_SIZE, BLOCK_SIZE, NBINS)

csvfile = open('features.csv', 'w')
fieldnames = ['id_image', 'features']
writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

# Saving AR on csv
for gradient in ar_gradients:
    filename = gradient[0]
    writer.writerow({'id_image': filename, 'features': gradient[1]})


# Saving NAR on csv
for gradient in nar_gradients:
    filename = gradient[0]
    writer.writerow({'id_image': filename, 'features': gradient[1]})
