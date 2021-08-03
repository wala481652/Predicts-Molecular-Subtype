import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from database import val, train
from ResNet50 import make_model

print("Tensorflow version " + tf.__version__)

BATCH_SIZE = 128
IMAGE_SIZE = (150, 150)
input_shape = (150, 150, 3)
NUM_EPOCHS = 100

train_database_patch = './database/dog_cat/train/'
validation_database_patch = './database/dog_cat/validation/'
train_database = train(train_database_patch, IMAGE_SIZE, BATCH_SIZE)
validation_database = val(validation_database_patch, IMAGE_SIZE, BATCH_SIZE)

# with strategy.scope():
model = make_model(input_shape)

callbacks = [
    ModelCheckpoint(
        "tensorflow/reunetdcm/cat_dog_model_resnet50.h5", save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4),
    CSVLogger("./database/CSV/GroundTruth.csv"),
    TensorBoard(),
    # EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False)
]


history = model.fit(train_database,
                    steps_per_epoch=train_database.samples // BATCH_SIZE,
                    validation_data=validation_database,
                    validation_steps=validation_database.samples // BATCH_SIZE,
                    epochs=NUM_EPOCHS,
                    callbacks=callbacks
                    )

model.save('cat_dog_model_resnet50')
