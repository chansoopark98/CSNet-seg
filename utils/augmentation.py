import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np


def color(x):
    """
         Color Augmentation
        :param x:  image inputs, 0~1
        :return: return images
        """
    x = tf.image.random_hue(x, 0.08)
    x = tf.image.random_saturation(x, 0.6, 1.6)
    x = tf.image.random_brightness(x, 0.05)
    x = tf.image.random_contrast(x, 0.7, 1.3)
    return x


def flip(x):
    """
        Flip image
        :param x:  image inputs, 0~1
        :return: return: images
        """
    x = tf.image.flip_left_right(x)  # 隨機左右翻轉影像
    return x


def zoom(x, scale_min=0.6, scale_max=1.6):
    """
        Zoom Image
        :param x:  image inputs, 0~1
        :param scale_min: minimum scale size
        :param scale_max: maximum scale size
        :return: return: images
        """
    h, w, _ = x.shape
    scale = tf.random.uniform([], scale_min, scale_max)
    nh = h * scale
    nw = w * scale
    x = tf.image.resize(x, (nh, nw))
    x = tf.image.resize_with_crop_or_pad(x, h, w)
    return x

def rotate(x, angle=(-45, 45)):
    """
        Rotate image
        :param angle: rotate angle
        :return: return: images
        """
    angle = tf.random.uniform([], angle[0], angle[1], tf.float32)
    theta = np.pi * angle / 180
    x = tfa.image.rotate(x, theta)
    return x