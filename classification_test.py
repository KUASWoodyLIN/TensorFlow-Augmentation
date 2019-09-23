import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from classification import parse_aug_fn


# Read Image
img_string = tf.io.read_file('./test/test_classification_image.jpg')
img = tf.image.decode_image(img_string)
img = img.numpy()

# Create dataset pipline
img_ds = tf.data.Dataset.from_tensor_slices([img])
label_ds = tf.data.Dataset.from_tensor_slices(['Meerkat'])
img_label_ds = tf.data.Dataset.zip({'image':img_ds, 'label':label_ds})
img_label_ds = img_label_ds.map(lambda dataset: parse_aug_fn(dataset, (416, 416)),
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)

h, w = (416, 416)
images = np.zeros((h * 4, w * 4, 3))
for count, [img, label] in enumerate(img_label_ds.repeat().take(16)):
    img = img.numpy()
    i = count // 4
    j = count % 4
    images[h * i:h * (i + 1), w * j:w * (j + 1)] = img
plt.figure(figsize=(12, 12))
plt.imshow(images)
plt.show()
