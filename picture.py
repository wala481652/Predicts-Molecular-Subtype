import numpy as np
import os
import csv
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet import preprocess_input
# from tensorflow.keras.callbacks import CSVLogger

csv_path = './database/CSV/picture.csv'
patch_path = './database/PNG/TCGA-CH-5740-01A-01-BS1'
patch_file = os.listdir(patch_path)
tabal = []

model = load_model('model_resnet50')

for i in range(len(patch_file)):

    img = image.load_img(patch_path + "/" + patch_file[i])
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    print('Predicted: ', preds)

    for p in preds:
        tabal.append(preds)
np.savetxt(csv_path, tabal, delimiter=",")
