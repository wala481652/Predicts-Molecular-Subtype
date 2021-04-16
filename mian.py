import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from database import val, train
from ResNet50 import make_model

print("Tensorflow version " + tf.__version__)

strategy = tf.distribute.MirroredStrategy()
print("Number of replicas:", strategy.num_replicas_in_sync)

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

BATCH_SIZE = 8
IMAGE_SIZE = (512, 512)
NUM_EPOCHS = 20

with strategy.scope():
    model = make_model()

callbacks = [
    ModelCheckpoint("tensorflow/reunetdcm/model.h5", save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4),
    CSVLogger("./database/CSV/GroundTruth.csv"),
    TensorBoard(),
    # EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False)
]


history = model.fit(train(),
                    steps_per_epoch=train().samples // BATCH_SIZE,
                    validation_data=val(),
                    validation_steps=val().samples // BATCH_SIZE,
                    epochs=NUM_EPOCHS,
                    callbacks=callbacks
                    )

model.save('model_resnet50')
