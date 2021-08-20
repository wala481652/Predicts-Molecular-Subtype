import re
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet import preprocess_input
# from tensorflow.keras.callbacks import CSVLogger

model_path = 'model_resnet50'
csv_path = './database/CSV/TCGA-CH-5740-01A-01-BS1_02.csv'
patch_path = './database/PNG/TCGA-CH-5740-01A-01-BS1/TCGA-CH-5740-01A-01-BS1_0_0.png'
patch_directory = './database/PNG/PRAD.1-ERG/TCGA-CH-5740-01A-01-BS1/'
patch_list = os.listdir(patch_directory)
table = []


def picture(patch_path):
    model = load_model(model_path)

    img = image.load_img(patch_path)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    return preds


if __name__ == '__main__':

    for i in range(len(patch_list)):
        preds = picture(patch_directory + patch_list[i])
        a = preds.reshape(2)
        table.append(a[1])

    np.savetxt(csv_path, table, delimiter=",")
