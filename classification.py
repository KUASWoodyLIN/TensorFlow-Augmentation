import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np


@tf.function
def parse_aug_fn(dataset, input_size=(416, 416), one_bot=False):
    """
    Image Augmentation function
    """
    ih, iw = input_size
    # (None, None, 3)
    x = tf.cast(dataset['image'], tf.float32) / 255.
    if one_bot:
        y = tf.one_hot(dataset['label'], 10)
    else:
        y = dataset['label']

    x = tf.image.resize(x, (ih, iw))

    # 觸發顏色轉換機率75%
    x = tf.cond(tf.random.uniform([], 0, 1) > 0.75, lambda: color(x), lambda: x)
    # 觸發影像翻轉機率50%
    x = tf.cond(tf.random.uniform([], 0, 1) > 0.5, lambda: flip(x), lambda: x)
    # 觸發影像縮放機率50%
    x = tf.cond(tf.random.uniform([], 0, 1) > 0.5, lambda: zoom(x), lambda: x)
    # 觸發影像旋轉機率50%
    x = tf.cond(tf.random.uniform([], 0, 1) > 0.5, lambda: rotate(x), lambda: x)
    return x, y


def color(x):
    """
         Color Augmentation

        :param x:  image inputs, 0~1
        :return: return images
        """
    x = tf.image.random_hue(x, 0.08)  # 隨機調整影像色調
    x = tf.image.random_saturation(x, 0.6, 1.6)  # 隨機調整影像飽和度
    x = tf.image.random_brightness(x, 0.05)  # 隨機調整影像亮度
    x = tf.image.random_contrast(x, 0.7, 1.3)  # 隨機調整影像對比度
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
    scale = tf.random.uniform([], scale_min, scale_max)  # 隨機縮放比例
    # 等比例縮放
    nh = h * scale # 縮放後影像長度
    nw = w * scale # 縮放後影像寬度
    x = tf.image.resize(x, (nh, nw))  # 影像縮放
    x = tf.image.resize_with_crop_or_pad(x, h, w)  # 影像裁減和填補
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