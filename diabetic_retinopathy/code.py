import cv2
import os
import numpy as np
import csv


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images


images = load_images_from_folder('./training/apparent_retinopathy/')

cell_size = (16, 16)
block_size = (4, 4)
nbins = 9

csvfile = open('features.csv', 'w')
for index, image in enumerate(images):
    # Tamanho de janela
    print('Imagem ' + str(index) + ' of ' + str(len(images)))
    win_size = (image.shape[1] // cell_size[1] * cell_size[1],
                image.shape[0] // cell_size[0] * cell_size[0])
    # Tamanho de Block
    hblock_size = (block_size[1] * cell_size[1], block_size[0] * cell_size[0])
    block_stride = (cell_size[1], cell_size[0])
    cell_size = (cell_size[1], cell_size[0])

    hog = cv2.HOGDescriptor(win_size, hblock_size, block_stride, cell_size, nbins)


    # compute(img[, winStride[, padding[, locations]]]) -> descriptors

    n_cells = (image.shape[0] // cell_size[0], image.shape[1] // cell_size[1])
    hog_feats = hog.compute(image)\
                .reshape(n_cells[1] - block_size[1] + 1,
                            n_cells[0] - block_size[0] + 1,
                            block_size[0], block_size[1], nbins) \
                .transpose((1, 0, 2, 3, 4))
    # hog.write('features.csv',)
    
    fieldnames = ['id_image', 'features']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerow({'id_image': index, 'features': hog})
    

# print(hog_feats)
# help(cv2.HOGDescriptor())
