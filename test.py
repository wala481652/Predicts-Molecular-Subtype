import tensorflow as tf
from tensorflow.keras import backend as K


def target_category_loss(x, category_index, nb_classes):
    return tf.mul(x, K.one_hot([category_index], nb_classes))


