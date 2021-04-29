import numpy as np
# from matplotlib import image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet import preprocess_input
# from database import *

patch_path = './database/PNG/norm/gleason4/03_46_57.png'
img = image.load_img(patch_path)
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

patch_path2 = './database/PNG/total/gleason4/03_46_57.png'
img = image.load_img(patch_path2)
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
y = preprocess_input(x)


model = load_model('model_resnet50')

preds1 = model.predict(x)
preds2 = model.predict(y)

print('Predicted: ', preds1, preds2)
