from tensorflow.keras.preprocessing.image import ImageDataGenerator


def train(data_file, IMAGE_SIZE, BATCH_SIZE):
    """
    data_file = 文件夾位子
    IMAGE_SIZE = 圖片大小
    BATCH_SIZE = 批次大小
    """
    train_datagen = ImageDataGenerator(featurewise_center=True,               # 使數據集去中心化（使得其均值為0）
                                       featurewise_std_normalization=True,    # 將输入除以數據標準差
                                       # rescale=1./255,                      # 將數值除與255 作正規化
                                       zca_whitening=False,                   # zca白化的作用是針對圖片進行PCA降維操作，減少圖片的冗餘信息，保留最重要的特徵
                                       horizontal_flip=True,                  # 隨機對圖片執行水 平翻轉操作
                                       vertical_flip=True,                    # 隨機對圖片執行上下翻轉操作
                                       channel_shift_range=10)
    train_batches = train_datagen.flow_from_directory(data_file,                         # 目标目录的路径。
                                                      target_size=IMAGE_SIZE,            # 所有的图像将被调整到的尺寸。
                                                      interpolation='bicubic',           # 在目标大小与加载图像的大小不同时，用于重新采样图像的插值方法。
                                                      # "categorical", "binary", "sparse", "input" 或 None 之一
                                                      # classes=['ERG', 'SPOP'],
                                                      class_mode='categorical',
                                                      # shuffle=True,                    # 是否混洗数据（默认 True）
                                                      batch_size=BATCH_SIZE              # 一批数据的大小（默认 32）。
                                                      )
    return train_batches


def val(data_file, IMAGE_SIZE, BATCH_SIZE):
    """
    data_file = 文件夾位子
    IMAGE_SIZE = 圖片大小
    BATCH_SIZE = 批次大小
    """
    valid_datagen = ImageDataGenerator(featurewise_center=True,               # 使數據集去中心化（使得其均值為0）
                                       featurewise_std_normalization=True,    # 将输入除以数据标准差
                                       # rescale=1./255,
                                       )
    valid_batches = valid_datagen.flow_from_directory(data_file,
                                                      target_size=IMAGE_SIZE,
                                                      interpolation='bicubic',
                                                      # classes=['ERG', 'SPOP'],
                                                      class_mode='categorical',
                                                      # shuffle=False,
                                                      batch_size=BATCH_SIZE)
    return valid_batches
