"""DataGenerator.py

This script contains the implmentation of class ImageGenerator which is used as a data generator
in the project nirschl_et_al

created by Jalil Ahmed.
"""
from src.WrapperFunctions import ceil, zeros, full, extract_patches, make_array, stack_vertical, rotate_array,\
    choose_random, make_categorical

from tensorflow.keras.utils import Sequence


class ImageGenerator(Sequence):
    """
    This class is a child class of the tensorflow.keras.Sequence class. It is used as a data generator in the project.
    The __getitem__() method of the class is especially written with the procd sject in mind.

    """
    def __init__(self, x, y, batch_size, num_patches, patch_size=64, num_classes=2, augmentation=False):
        """ Constructor of the class ImageGenerator

        :param x: Array of images
        :type x: numpy.ndarray
        :param y: Array of labels
        :type y: numpy.ndarray
        :param batch_size: Size of the batch to be generated
        :type batch_size: int
        :param num_patches: Number of patches to be extracted from each sample image
        :type num_patches: int
        :param patch_size: (Default=64) Size of the patch
        :type patch_size: int
        :param num_classes: (Default=2) Number of classes of images
        :type num_classes: int
        :param augmentation: (Default=False) Toggle for data augmentation. If True then data is augmented with rotations
        :type augmentation: Bool
        """
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.augmentation_toggle = augmentation

    def __len__(self):
        """ This function calculates the number of batches to be made

        :return: Number of batches
        :rtype: int
        """
        return ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        """ This function is called by the model to extract samples and labels

        :param idx: Index at which the Image generator is
        :return: A tuple of samples and one-hot-encoded labels
        :rtype: (numpy.ndarray, numpy.ndarray)
        """
        temp_x = self.x[idx*self.batch_size: (idx + 1)*self.batch_size]
        temp_y = self.y[idx*self.batch_size: (idx + 1)*self.batch_size]

        num_samples = int(self.num_patches*len(temp_x))

        batch_x = zeros((num_samples, self.patch_size, self.patch_size, 3))
        batch_y = zeros((num_samples, 1))

        for i, (sample, label) in enumerate(zip(temp_x, temp_y)):
            batch_x[i*self.num_patches: i*self.num_patches+self.num_patches, :] = extract_patches(sample,
                                                                                                  patch_size=
                                                                                                  self.patch_size,
                                                                                                  num_patches=
                                                                                                  self.num_patches)
            batch_y[i*self.num_patches: i*self.num_patches+self.num_patches, :] = full((self.num_patches, 1), label)

        if self.augmentation_toggle:
            augmented_x = make_array([rotate_array(x, choose_random([90, 180, 360])) for x in batch_x])
            batch_x = stack_vertical((batch_x, augmented_x))
            batch_y = stack_vertical((batch_y, batch_y))

        return batch_x, make_categorical(batch_y, self.num_classes)