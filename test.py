# -*- coding: UTF-8 -*-
import tensorflow as tf
import IPython.display as display
import numpy as np

cat_in_snow = tf.keras.utils.get_file('320px-Felis_catus-cat_on_snow.jpg',
                                      'https://storage.googleapis.com/download.tensorflow.org/example_images/320px-Felis_catus-cat_on_snow.jpg')
williamsburg_bridge = tf.keras.utils.get_file('194px-New_East_River_Bridge_from_Brooklyn_det.4a09796u.jpg',
                                              'https://storage.googleapis.com/download.tensorflow.org/example_images/194px-New_East_River_Bridge_from_Brooklyn_det.4a09796u.jpg')

display.display(display.Image(filename=cat_in_snow))
display.display(display.HTML(
    'Image cc-by: &lt;a "href=https://commons.wikimedia.org/wiki/File:Felis_catus-cat_on_snow.jpg"&gt;Von.grzanka&lt;/a&gt;'))
display.display(display.Image(filename=williamsburg_bridge))
display.display(display.HTML(
    '&lt;a "href=https://commons.wikimedia.org/wiki/File:New_East_River_Bridge_from_Brooklyn_det.4a09796u.jpg"&gt;From Wikimedia&lt;/a&gt;'))

# The following functions can be used to convert a value to a type compatible
# with tf.Example.


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        # BytesList won't unpack a string from an EagerTensor.
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


"""
和以前一样，将特征编码为与 tf.Example 兼容的类型。这将存储原始图像字符串特征，以及高度、宽度、深度和任意 label 特征。
后者会在您写入文件以区分猫和桥的图像时使用。将 0 用于猫的图像，将 1 用于桥的图像：
"""
image_labels = {
    cat_in_snow: 0,
    williamsburg_bridge: 1,
}


# 這是一個示例，僅使用 cat 圖像。
image_string = open(cat_in_snow, 'rb').read()
label = image_labels[cat_in_snow]


# 創建一個字典，其中包含可能相關的特徵。
def image_example(image_string, label):
    image_shape = tf.image.decode_jpeg(image_string).shape

    feature = {
        'height': _int64_feature(image_shape[0]),
        'width': _int64_feature(image_shape[1]),
        'depth': _int64_feature(image_shape[2]),
        'label': _int64_feature(label),
        'image_raw': _bytes_feature(image_string),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


for line in str(image_example(image_string, label)).split('\n')[:15]:
    print(line)
print('...')

# Write the raw image files to `images.tfrecords`.
# First, process the two images into `tf.Example` messages.
# Then, write to a `.tfrecords` file.
record_file = "./images.tfrecords"
with tf.io.TFRecordWriter(record_file) as writer:
    for filename, label in image_labels.items():
        image_string = open(filename, 'rb').read()
        tf_example = image_example(image_string, label)
        writer.write(tf_example.SerializeToString())
