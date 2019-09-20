import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np


@tf.function
def parse_aug_fn(dataset, input_size=(416, 416)):
    """
    Image Augmentation function
    """
    ih, iw = input_size
    # (None, None, 3)
    x = tf.cast(dataset['image'], tf.float32)
    # (y1, x1, y2, x2, class)
    bbox = dataset['objects']['bbox']
    label = dataset['objects']['label']

    x, bbox = resize(x, bbox, input_size)

    # Color Augmentation 75%
    x = tf.cond(tf.random.uniform([], 0, 1) > 0.75, lambda: color(x), lambda: x)
    # Flip Image 50%
    x, bbox = tf.cond(tf.random.uniform([], 0, 1) > 0.5, lambda: flip(x, bbox), lambda: (x, bbox))
    # Zoom Image 50%
    x, bbox, label = tf.cond(tf.random.uniform([], 0, 1) > 0.5, lambda: zoom(x, bbox, label), lambda: (x, bbox, label))
    # Rotate Image 50%
    x, bbox, label = tf.cond(tf.random.uniform([], 0, 1) > 0.5, lambda: rotate(x, bbox, label), lambda: (x, bbox, label))

    # normalization
    x = x / 255.

    # list [x1, y1, x2, y2] -> tensor [y1, x1, y2, x2]
    bbox = tf.stack([bbox[1], bbox[0], bbox[3], bbox[2]], axis=-1)
    bbox = tf.divide(bbox, [ih, iw, ih, iw])
    return x, bbox, label


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


def resize(x, bboxes, input_size):
    """
        Resize Image

        :param x:  image inputs, 0~1
        :param bboxes: bounding boxes inputs, shape(x, 4) "y1, x1, y2, x2", scale 0~1
        :param input_size: Network input size
        :return: return (images, bboxes),
                    images: return scale0~1,
                    bboxes: return list [y1, x1, y2, x2], scale 0~w, 0~h
        """
    ih, iw = input_size
    # image resize
    x = tf.image.resize(x, (ih, iw))

    # bounding box resize
    bboxes = tf.multiply(bboxes, [ih, iw, ih, iw])
    y1 = bboxes[..., 0]
    x1 = bboxes[..., 1]
    y2 = bboxes[..., 2]
    x2 = bboxes[..., 3]
    return x, [y1, x1, y2, x2]


def flip(x, bboxes):
    """
        Flip image

        :param x:  image inputs, 0~1
        :param bboxes: bounding boxes inputs list [y1, x1, y2, x2], scale 0~w, 0~h
        :return: return (images, bboxes),
                    images: return scale0~1,
                    bboxes: return list [y1, x1, y2, x2], scale 0~w, 0~h
        """
    h, w, c = x.shape
    x = tf.image.flip_left_right(x)  # 隨機左右翻轉影像
    y1 = bboxes[0]
    x1 = w - bboxes[3]
    y2 = bboxes[2]
    x2 = w - bboxes[1]
    return x, [y1, x1, y2, x2]


def zoom(x, bboxes, label, scale_min=0.6, scale_max=1.6):
    """
        Zoom Image

        :param x:  image inputs, 0~1
        :param bboxes: bounding boxes inputs list [y1, x1, y2, x2], scale 0~w, 0~h
        :param scale_min: minimum scale size
        :param scale_max: maximum scale size
        :return: return (images, bboxes),
                    images: return scale0~1,
                    bboxes: return list [y1, x1, y2, x2], scale 0~w, 0~h
        """
    h, w, _ = x.shape
    scale = tf.random.uniform([], scale_min, scale_max)  # 隨機縮放比例
    # 等比例縮放
    nh = tf.cast(h * scale, tf.int32)  # 縮放後影像長度
    nw = tf.cast(w * scale, tf.int32)  # 縮放後影像寬度

    # 如果將影像縮小執行以下程式
    def scale_less_then_one():
        resize_x = tf.image.resize(x, (nh, nw))  # 影像縮放
        dy = tf.random.uniform([], 0, (h - nh), tf.int32)
        dx = tf.random.uniform([], 0, (w - nw), tf.int32)
        indexes = tf.meshgrid(tf.range(dy, dy+nh), tf.range(dx, dx+nw), indexing='ij')
        indexes = tf.stack(indexes, axis=-1)
        output = tf.scatter_nd(indexes, resize_x, (h, w, 3))
        return output, dx, dy

    # 如果將影像放大執行以下以下程式
    def scale_greater_then_one():
        resize_x = tf.image.resize(x, (nh, nw))  # 影像縮放
        dy = tf.random.uniform([], 0, (nh - h), tf.int32)
        dx = tf.random.uniform([], 0, (nw - w), tf.int32)
        return resize_x[dy:dy + h, dx:dx + w], -dx, -dy

    def scale_equal_zero():
        return x, 0, 0

    output, dx, dy = tf.case([(tf.logical_or(tf.less(nh - h, 0), tf.less(nw - w, 0)), scale_less_then_one),
                              (tf.logical_or(tf.greater(nh - h, 0), tf.greater(nw - w, 0)), scale_greater_then_one)],
                             default=scale_equal_zero)

    # 重新調整bounding box位置
    y1 = bboxes[0] * scale + tf.cast(dy, dtype=tf.float32)
    x1 = bboxes[1] * scale + tf.cast(dx, dtype=tf.float32)
    y2 = bboxes[2] * scale + tf.cast(dy, dtype=tf.float32)
    x2 = bboxes[3] * scale + tf.cast(dx, dtype=tf.float32)
    # 如果座標超出範圍將其限制在邊界上
    y1 = tf.where(y1 < 0, tf.zeros_like(y1), y1)
    x1 = tf.where(x1 < 0, tf.zeros_like(x1), x1)
    y2 = tf.where(y2 > h, h * tf.ones_like(y2), y2)
    x2 = tf.where(x2 > w, w * tf.ones_like(x2), x2)
    # 找出不存在影像上的bounding box並剔除
    box_w = x2 - x1
    box_h = y2 - y1
    bboxes_filter = tf.logical_and(box_w > 1, box_h > 1)
    y1 = y1[bboxes_filter]
    x1 = x1[bboxes_filter]
    y2 = y2[bboxes_filter]
    x2 = x2[bboxes_filter]
    label = label[bboxes_filter]
    output = tf.ensure_shape(output, x.shape)
    return output, [y1, x1, y2, x2], label


def rotate(img, bboxes, label, angle=(-45, 45)):
    """
        Rotate image

        :param img: image inputs, 0~1
        :param bboxes: bounding boxes inputs list [y1, x1, y2, x2], 0~w, 0~h
        :param label: bounding boxes label
        :param angle: rotate angle
        :return: return (output, bboxes, label),
                    output: return images: scale0~1,
                    bboxes: return list [y1, x1, y2, x2], scale 0~w, 0~h
                    label: return bounding boxes label
        """
    h, w, c = img.shape
    print(img.shape)
    cx, cy = w // 2, h // 2
    angle = tf.random.uniform([], angle[0], angle[1], tf.float32)

    theta = np.pi * angle / 180
    output = tfa.image.rotate(img, theta)
    # convert (ymin, xmin, ymax, xmax) to corners
    width = bboxes[3] - bboxes[1]
    height = bboxes[2] - bboxes[0]
    x1 = bboxes[1]
    y1 = bboxes[0]
    x2 = x1 + width
    y2 = y1
    x3 = x1
    y3 = y1 + height
    x4 = bboxes[3]
    y4 = bboxes[2]
    corners = tf.stack((x1, y1, x2, y2, x3, y3, x4, y4), axis=-1)

    # calculate the rotate bboxes
    corners = tf.reshape(corners, (-1, 2))
    corners = tf.concat((corners, tf.ones((tf.shape(corners)[0], 1), dtype=corners.dtype)), axis=-1)

    alpha = tf.cos(theta)
    beta = tf.sin(theta)
    M = tf.reshape(tf.stack([alpha, beta, (1 - alpha) * cx - beta * cy, -beta, alpha, beta * cx + (1 - alpha) * cy]),
                   (2, 3))
    corners = tf.matmul(corners, M, transpose_b=True)
    corners = tf.reshape(corners, (-1, 8))

    # convert corners to (xmin, ymin, xmax, ymax)
    x_ = corners[:, ::2]
    y_ = corners[:, 1::2]
    x1 = tf.reduce_min(x_, axis=-1)
    y1 = tf.reduce_min(y_, axis=-1)
    x2 = tf.reduce_max(x_, axis=-1)
    y2 = tf.reduce_max(y_, axis=-1)

    # 如果座標超出範圍將其限制在邊界上
    y1 = tf.where(y1 < 0, tf.zeros_like(y1), y1)
    x1 = tf.where(x1 < 0, tf.zeros_like(x1), x1)
    y2 = tf.where(y2 > h, h * tf.ones_like(y2), y2)
    x2 = tf.where(x2 > w, w * tf.ones_like(x2), x2)

    # 找出不存在影像上的bounding box並剔除
    box_w = x2 - x1
    box_h = y2 - y1
    bboxes_filter = tf.logical_and(box_w > 1, box_h > 1)
    y1 = y1[bboxes_filter]
    x1 = x1[bboxes_filter]
    y2 = y2[bboxes_filter]
    x2 = x2[bboxes_filter]
    label = label[bboxes_filter]
    return output, [y1, x1, y2, x2], label