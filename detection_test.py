import tensorflow as tf
# import tensorflow_addons as tfa
import numpy as np
import cv2
import matplotlib.pyplot as plt
from detection import parse_aug_fn

# Read Image
img_string = tf.io.read_file('test/test_detection_image.jpg')
img = tf.image.decode_image(img_string)
img = img.numpy()

# Read Bounding box
with open('test/test_detection_label.txt', 'r') as file:
    lines = file.readlines()
    bboxes, labels = [], []
    for line in lines:
        bboxes.append([float(box) for box in line.split()[:-1]])
        labels.append(int(line.split()[-1]))
classes_list = ['bottle', 'tvmonitor', 'person']

# Draw Image and Bounding bix
draw_img = img.copy()
h, w, _ = draw_img.shape
for bbox, label in zip(bboxes, labels):
    y1 = int(bbox[0] * h)
    x1 = int(bbox[1] * w)
    y2 = int(bbox[2] * h)
    x2 = int(bbox[3] * w)
    cv2.rectangle(draw_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(draw_img,
                classes_list[label],
                (x1, y1 - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 244, 0), 2)
plt.imshow(draw_img)


# Create dataset pipline
img_ds = tf.data.Dataset.from_tensor_slices([img])
bbox_ds = tf.data.Dataset.from_tensor_slices([bboxes])
label_ds = tf.data.Dataset.from_tensor_slices([labels])
objects_ds = tf.data.Dataset.zip({'bbox':bbox_ds, 'label':label_ds})
img_objects_ds = tf.data.Dataset.zip({'image':img_ds, 'objects':objects_ds})
img_objects_ds = img_objects_ds.map(lambda dataset: parse_aug_fn(dataset, (416, 416)),
                                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
h, w = (416, 416)
images = np.zeros((h * 4, w * 4, 3))
for count, [img, bboxes] in enumerate(img_objects_ds.repeat().take(16)):
    bboxes = tf.multiply(bboxes, [h, w, h, w, 1])
    img = img.numpy()
    box_indices = tf.where(tf.reduce_sum(bboxes, axis=-1))
    bboxes = tf.gather_nd(bboxes, box_indices)
    for box, label in zip(bboxes, labels):
        x1 = tf.cast(box[0], tf.int16).numpy()
        y1 = tf.cast(box[1], tf.int16).numpy()
        x2 = tf.cast(box[2], tf.int16).numpy()
        y2 = tf.cast(box[3], tf.int16).numpy()
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 1, 0), 2)
        cv2.putText(img,
                    classes_list[box[4]],
                    (x1, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 1, 0), 2)

    i = count // 4
    j = count % 4
    images[h * i:h * (i + 1), w * j:w * (j + 1)] = img
plt.figure(figsize=(12, 12))
plt.imshow(images)
plt.show()
