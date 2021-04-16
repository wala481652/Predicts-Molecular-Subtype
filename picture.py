import numpy as np
import tensorflow as tf
from matplotlib import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet import decode_predictions
# from tensorflow.keras.preprocessing import image
from database import *

# patch_path = './database/PNG/train/gleason5/01_12_42.png'
# image = image.imread(patch_path)
# img = image.load_img(patch_path)
# x = image.img_to_array(img)
# x = np.expand_dims(image, axis=0)
# x = tf.keras.applications.resnet.preprocess_input(x)

model = load_model('model_resnet50')

preds = model.predict(val())

print('Predicted: ', preds)
