import cv2


class SVM:
    def __init__(self, cell_size, block_size, n_bins):
        self.cell_size = cell_size
        self.block_size = block_size
        self.n_bins = n_bins

    def hog(self, image, cell_size, block_size, n_bins):
        win_size = (image.shape[1] // cell_size[1] * cell_size[1],
                    image.shape[0] // cell_size[0] * cell_size[0])
        block = (block_size[1] * cell_size[1], block_size[0] * cell_size[0])
        stride = (cell_size[1], cell_size[0])
        cell_size = (cell_size[1], cell_size[0])
        return cv2.HOGDescriptor(win_size, block, stride, cell_size, n_bins)
