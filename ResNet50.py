import tensorflow as tf
# from tensorflow.python.keras.backend import flatten
# import tensorflow_addons as tfa
from tensorflow.keras.applications import ResNet50


def make_model(input_shape):


    learning_rate = 1e-4

    base_model = ResNet50(include_top=False, weights=None,
                          input_shape=input_shape)
    # base_model.trainable = True  # 凍結Resnet的權重

    model = tf.keras.Sequential([
        base_model,
        # tfa.layers.GroupNormalization(groups=2, axis=3),
        tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.Flatten(),
        # tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        # tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    return model
