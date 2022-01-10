"""Wrapper Functions

This script contains the methods which wrap the various methods provided by different libraries.
These wrapper methods are written to facilitate future changes.

created by Jalil Ahmed.
"""

import math
import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d
from scipy.ndimage import rotate
from sklearn.utils import shuffle
from random import choice
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import cv2
from src.GlobalVariables import RANDOM_SEED


def ceil(float_value):
    """ Returns the ceil value of a float

    :param float_value: Value whose ceil is to be returned
    :type float_value: float
    :return: Ceil value of the float
    :rtype: int
    """

    return math.ceil(float_value)


def zeros(size):
    """ Return an numpy array with zeros of size

    :param size: Size of numpy array
    :type size: tuple
    :return: A numpy array with zeros
    :rtype: numpy.ndarray
    """

    return np.zeros(size)


def extract_patches(image, patch_size, num_patches):
    """ Divide an image to patches

    :param image: Input image
    :type image: numpy.ndarray
    :param patch_size: Size of a patches
    :type patch_size: int
    :param num_patches: Number of patches to be extracted
    :type num_patches: int
    :return: A numpy array containing patches of the image
    :rtype: numpy.ndarray
    """

    image_patches = extract_patches_2d(image, (patch_size, patch_size), max_patches=num_patches,
                                       random_state=RANDOM_SEED)
    return image_patches


def full(size, value):
    """ Create a numpy array filled with a constant value

    :param size: Size of numpy array
    :type size: tuple
    :param value: Value to fill the numpy array
    :type value: int
    :return: A numpy array
    :rtype: numpy.ndarray
    """

    return np.full(size, value)


def make_array(input_list):
    """ Convert a list to numpy array

    :param input_list: A list
    :type input_list: list
    :return: A numpy array
    :rtype: numpy.ndarray
    """

    return np.array(input_list)


def stack_vertical(input_tuple):
    """ Vertically stack a tuple of numpy array

    :param input_tuple: A tuple containing numpy arrays
    :type input_tuple: (numpy.ndarray, numpy.ndarray)
    :return: A numpy array
    :rtype: numpy.ndarray
    """

    return np.vstack(input_tuple)


def rotate_array(input_array, rotation_angle):
    """A wrapper function of the scipy rotate function. This returns the array rotated with rotation angle
    at the default axis=1

    :param input_array: A numpy array
    :type input_array: numpy.ndarray
    :param rotation_angle: Angle by which to rotate the array
    :type rotation_angle: int
    :return: A numpy array
    :rtype: numpy.ndarray
    """

    return rotate(input_array, rotation_angle)


def choose_random(input_list):
    """ Choose a value from a list at random

    :param input_list: A list of values
    :type input_list: list
    :return: A single value from input list
    :rtype: int or float
    """

    return choice(input_list)


def make_categorical(labels, num_classes):
    """ One hot encode an array of prediction labels. It is wrapper function to the to_categorical function in keras

    :param labels: A numpy array of labels
    :type labels: numpy.ndarray
    :param num_classes: Number of classes
    :type num_classes: int
    :return: One-hot encoded array of labels
    :rtype: numpy.ndarray
    """

    return to_categorical(labels, num_classes=num_classes)


def convert_color(image, gray_scale_toggle):
    """ Convert the channels of images because CV2 library channels are BGR

    :param image: A numpy array of image with channel configuration BGR
    :type image: numpy.ndarray
    :param gray_scale_toggle: Toggle for grayscale output
    :type gray_scale_toggle: Bool
    :return: A numpy array of image with channel configuration RGB or Gray
    :rtype: numpy.ndarray
    """

    if gray_scale_toggle:
        output_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float)
    else:
        output_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float)
    return output_image


def imread(image_path):
    """ Read image from image path. This is a wrapper function of cv2.imread()

    :param image_path: Path of the image
    :type image_path: str
    :return: A numpy array of the image
    :rtype: numpy.ndarray
    """

    return cv2.imread(image_path, 1)


def shuffle_data(images, labels):
    """ Function to shuffle data i.e. images and labels. A wrapper function of sklearn.utils shuffle

    :param images: A numpy array of images
    :type images: numpy.ndarray
    :param labels: A numpy array of labels
    :type labels: numpy.ndarray
    :return: Shuffle images and labels
    :rtype: (numpy.ndarray, numpy.ndarray)
    """

    return shuffle(images, labels, random_state=RANDOM_SEED)


def split_data(images, labels, test_size):
    """ To split the data set into train and test sets stratified with labels.
    This is a wrapper function to sklearn.model_selection.train_test_split

    :param images: Array of images
    :type images: numpy.ndarray
    :param labels: Array of labels
    :type labels: numpy.ndarray
    :param test_size: Size of the test set
    :type test_size: float
    :return: Train and test split of samples and labels
    :rtype: (numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray)
    """

    return train_test_split(images, labels, random_state=RANDOM_SEED, stratify=labels, test_size=test_size)