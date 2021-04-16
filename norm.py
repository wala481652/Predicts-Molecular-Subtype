import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
from sklearn.cluster import MiniBatchKMeans


def norm(path, K):
    # 讀取圖片
    ima = image.imread(path)
    w, h, d = tuple(ima.shape)
    image_data = np.reshape(ima, (w * h, d))

    # 將顏色分類為 K 種
    kmeans = MiniBatchKMeans(n_clusters=K, batch_size=5000)
    labels = kmeans.fit_predict(image_data)
    centers = kmeans.cluster_centers_

    # 根據分類將顏色寫入新的影像陣列
    image_compressed = np.zeros(ima.shape)
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image_compressed[i][j] = centers[labels[label_idx]]
            label_idx += 1

    return image_compressed


if __name__ == '__main__':
    path = './database/PNG/test/01_12_42.png'
    ima = image.imread(path)
    K = 3
    # 如果想儲存壓縮後的圖片, 將下面這句註解拿掉
    #plt.imsave(r'C:\Users\使用者名稱\Downloads\compressed.jpg', image_compressed)
    # 顯示原圖跟壓縮圖的對照
    plt.figure(figsize=(12, 9))
    plt.subplot(121)
    plt.title('Original photo')
    plt.imshow(ima)
    plt.subplot(122)
    plt.title(f'Compressed to KMeans={K} colors')
    plt.imshow(norm(path, K))
    plt.tight_layout()
    plt.show()
