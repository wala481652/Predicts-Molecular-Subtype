import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from database import val, train
from ResNet50 import make_model

print("Tensorflow version " + tf.__version__)

cross_device_ops = tf.distribute.HierarchicalCopyAllReduce()
strategy = tf.distribute.MirroredStrategy(cross_device_ops=cross_device_ops)
print("Number of replicas:", strategy.num_replicas_in_sync)

BATCH_SIZE = 32
IMAGE_SIZE = (512, 512)
input_shape = (512, 512, 3)
NUM_EPOCHS = 100

train_database_patch = './database/train/'
train_database = train(train_database_patch, IMAGE_SIZE, BATCH_SIZE)
validation_database = val(train_database_patch, IMAGE_SIZE, BATCH_SIZE)

AUC = tf.keras.metrics.AUC(name='auc', multi_label=True)

callbacks = [
    ModelCheckpoint("tensorflow/reunetdcm/model.h5", save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4),
    CSVLogger("./database/CSV/GroundTruth.csv"),
    TensorBoard(),
    # EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False)
]

with strategy.scope():
    model = make_model(input_shape)

    model.summary()

    model.compile(
        optimizer='adam',
        # loss="BinaryCrossentropy",
        loss="categorical_crossentropy",
        metrics=['accuracy']
    )

    model.summary()

    history = model.fit(train_database,
                        steps_per_epoch=train_database.samples // BATCH_SIZE,
                        validation_data=validation_database,
                        validation_steps=validation_database.samples // BATCH_SIZE,
                        epochs=NUM_EPOCHS,
                        callbacks=callbacks
                        )

    model.save('model_resnet50')
