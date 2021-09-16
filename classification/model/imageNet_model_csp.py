import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Add, BatchNormalization, Activation

def conv3x3(x, out_planes, strides=1):
    """
    Standard Convolution without bias
    :param x: input tensor
    :param out_planes: output filters
    :return: 4D tensor
    """

    return Conv2D(filters=out_planes, kernel_size=(3, 3), strides=strides, padding='same', bias=False)(x)



