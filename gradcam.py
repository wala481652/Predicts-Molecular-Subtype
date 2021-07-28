import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tensorflow import keras as keras
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet import preprocess_input


def get_img_array(img_path, size):
    """
    img_path : 圖片位址
    size : 圖片大小
    """
    # 'img'是大小為512x512的PIL圖片
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    # 'array'是一個float32 Numpy形狀的數組（512，512，3）
    array = keras.preprocessing.image.img_to_array(img)
    # 我們添加了一個維，以將數組轉換為大小為（1、299、299、3）的'batch'
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    生成熱力圖
    """
    # 首先，我們創建一個模型，將輸入圖像映射到最後一個conv層的激活以及輸出預測
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])

    # 然後，我們針對最後一個conv層的激活計算輸入圖像的頂級預測類的梯度
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # 這是輸出神經元（最後預測或選擇的）相對於最後一個conv層的輸出特徵圖的梯度
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # 這是一個向量，其中每個條目都是特定特徵圖通道上的梯度的平均強度
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 我們將特徵圖數組中的每個通道乘以相對於頂級預測類的“該通道的重要性”，
    # 然後將所有通道求和以獲得熱圖類激活
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # 為了可視化，我們還將正規化0和1之間的熱圖
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()


def save_and_display_gradcam(img_path, heatmap, alpha=0.4):
    # 加載原始圖像
    img = keras.preprocessing.image.load_img(img_path)
    img = keras.preprocessing.image.img_to_array(img)

    # 將熱圖重新縮放到0-255的範圍
    heatmap = np.uint8(255 * heatmap)

    # 使用色圖為熱圖著色
    jet = cm.get_cmap("jet")

    # 使用色圖的RGB值的arange
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # 使用RGB彩色熱圖創建圖像
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # 將熱圖疊加在原始圖像上
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    # 保存疊加的圖像
    # plt.imshow(superimposed_img)
    # plt.show()
    # superimposed_img.save(cam_path)

    return superimposed_img


if __name__ == '__main__':
    patch_path = './database/PNG/PRAD.1-ERG/TCGA-KK-A6E1-01Z-00-DX1_89_137.png'
    img_size = (512, 512)
    img_array = preprocess_input(get_img_array(patch_path, size=img_size))
    last_conv_layer_name = "group_normalization"

    model = load_model('model_resnet50')
    model.summary()

    # 刪除最後一層的softmax
    model.layers[-1].activation = None

    # 打印最高預測類別
    preds = model.predict(img_array)

    # 生成類激活熱圖
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    # plt.imshow(heatmap)
    # plt.show()
    # plt.imsave('./database/PNG/TCGA-CH-5739-01A-01-BS1_26_20_heatmap.png', heatmap)

    heatmap_img = save_and_display_gradcam(patch_path, heatmap)
    plt.imshow(heatmap_img)
    plt.show()
    # plt.imsave('./database/PNG/heatmap_img.pn-g', heatmap_img)
    heatmap_img.save('./database/PNG/TCGA-KK-A6E1-01Z-00-DX1_89_137_heatmap_img.png')
