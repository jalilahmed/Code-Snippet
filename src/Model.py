""" Model

This script implement the model architecture as published in the publication:

A deep-learning classifier identifies patients with clinical heart failure using whole-slide images of H&E tissue
Nirschl JJ, Janowczyk A, Peyster EG, Frank R, Margulies KB, et al. (2018)
PLOS ONE 13(4): e0192726. https://doi.org/10.1371/journal.pone.0192726

created by Jalil Ahmed.
"""
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Softmax
from .ChannelAverageLayer import ChannelAverage


def conv_block(input_tensor, num_kernels, kernel_size, stride):
    """ Create a block of a Convolutional, Batch Normalization, and Activation function layers.

    :param input_tensor: Input tensor of features
    :type input_tensor: tensorflow.Tensor
    :param num_kernels: Number of kernels in the convolutional layer
    :type num_kernels: int
    :param kernel_size: Size of each kernel
    :type kernel_size: int
    :param stride: Stride of the convolutional layer
    :type stride: int
    :return: A tensorflow tensor
    :rtype: tensorflow.Tensor
    """

    x = Conv2D(num_kernels, kernel_size, stride, padding='valid', kernel_regularizer='l2')(input_tensor)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x


def create_model(input_shape=(64, 64, 3), num_classes=2, model_name='MODEL_NAME'):
    """ Create a model based upon the architecture in the publication.

    :param input_shape: Shape of the input image
    :type input_shape: (int, int, int)
    :param num_classes: (Default=2) Number of classes
    :type num_classes: int
    :param model_name: Name of the model
    :type model_name: string
    :return: Convolutional Neural Network
    :rtype: tensorflow.keras.Model
    """

    input_tensor = Input(shape=input_shape, name='input_layer')
    x = conv_block(input_tensor, 16, 3, 1)
    x = conv_block(x, 16, 2, 2)
    x = conv_block(x, 16, 3, 1)
    x = conv_block(x, 16, 3, 2)
    x = conv_block(x, 16, 3, 1)
    x = conv_block(x, 16, 4, 2)
    x = conv_block(x, num_classes, 5, 1)
    x = ChannelAverage()(x)
    output_tensor = Softmax(axis=-1)(x)

    model = Model(input_tensor, output_tensor, name=model_name)

    return model
