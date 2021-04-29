import tensorflow as tf
# from tensorflow.python.keras.backend import flatten
import tensorflow_addons as tfa
from tensorflow.keras.applications import ResNet50


def make_model():
    # metrics = [tf.keras.metrics.AUC(name='auc', multi_label=True)]
    learning_rate = 1e-4

    base_model = ResNet50(include_top=False, weights=None,
                          input_shape=(512, 512, 3))
    base_model.trainable = True  # 凍結Resnet的權重

    model = tf.keras.Sequential([
        base_model,
        tfa.layers.GroupNormalization(groups=2, axis=3),
        # tf.keras.layers.Flatten(),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(4, activation='softmax')])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=['accuracy']
    )
    model.summary()
    return model
