import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet import preprocess_input
# from tensorflow.keras.callbacks import CSVLogger

csv_path = './database/CSV/TCGA-CH-5788-01A-01-BS1_5_12.csv'
# patch_path = './database/PNG/TCGA-CH-5740-01A-01-BS1/TCGA-CH-5740-01A-01-BS1_0_0.png'
patch_path = './database/PNG/PRAD.5-SPOP/TCGA-CH-5788-01A-01-BS1/'
patch_file = os.listdir(patch_path)
table = []

model = load_model('./1100802_model_resnet50')

for i in range(len(patch_file)):
    img = image.load_img(patch_path + patch_file[i])
    # img = image.load_img(patch_path)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    a = preds.reshape(2)
    print('Predicted: ', a)

    table.append(a[0])

np.savetxt(csv_path, table, delimiter=",")
