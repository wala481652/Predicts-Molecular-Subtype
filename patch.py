import openslide as ops
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
from sklearn.cluster import MiniBatchKMeans

os.environ['PATH'] = "./lib/openslide-win64-20171122/bin" + \
    ";" + os.environ['PATH']
print(os.environ['PATH'])


def kmeans(path, K):
    """
    使用k-means對圖片做normlization

    path=圖片位址,
    k=k群數量
    """
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


def SVS_to_PNGPatch(svs_file):
    """
    將SVS切成數張512*512的patch

    svs_file=svs檔案位址
    """
    # 去除副檔名
    file_basename = os.path.basename(svs_file).split('.')[0]

    # 建立目錄
    if os.path.exists('./database/PNG/PRAD.1-ERG/' + file_basename) == False:
        os.mkdir('./database/PNG/PRAD.1-ERG/' + file_basename)
        print(file_basename+"已建立")
    else:
        print(file_basename+"已存在")

    # 讀取SVS
    slide = ops.OpenSlide(svs_file)

    # 讀取SVS的長寬及Patch的數量
    [w, h] = slide.level_dimensions[0]
    print(w, h)
    [N, M] = [w/512, h/512]
    print(N, M)

    # 儲存Patch為PNG
    for i in range(int(N)):
        for j in range(int(M)):
            region = slide.read_region((i*512, j*512), 0, (512, 512))

            if i < 10:
                x = '00'+str(i)
            elif i < 100:
                x = '0'+str(i)
            else:
                x = str(i)

            if j < 10:
                y = '00'+str(j)
            elif j < 100:
                y = '0'+str(j)
            else:
                y = str(j)

            region.save('./database/PNG/PRAD.1-ERG/' + file_basename +
                        '/' + file_basename + '_' + x + '_' + y + '.png')
            print(file_basename + '_' + x + '_' + y + '.png')

    return region


if __name__ == '__main__':
    svs_directory = './database/TCGA Molecular Subtype/PRAD.1-ERG/'
    svs_file_list = os.listdir(svs_directory)

    # svs_file = './database/TCGA Molecular Subtype/PRAD.1-ERG/TCGA-CH-5788-01A-01-BS1.c4e36368-30be-4dc8-9931-279afeeea401.svs'
    # SVS_to_PNGPatch(svs_file)

    # patch_directory = './database/PNG/TCGA Molecular Subtype/PRAD.1-ERG/'
    # patach_file_list = os.listdir(patch_directory)
    # K = 3

    # patch_file = './database/PNG/test/01_12_42.png'
    # ima = image.imread(patch_file)
    # norm_img = kmeans(patch_file, K)

    # print(norm_img.shape)
    # # 顯示原圖跟壓縮圖的對照
    # plt.figure(figsize=(12, 9))
    # plt.subplot(121)
    # plt.title('Original photo')
    # plt.imshow(ima)
    # plt.subplot(122)
    # plt.title(f'Compressed to KMeans={K} colors')
    # plt.imshow(norm_img)
    # plt.tight_layout()
    # plt.show()

    for i in range(len(svs_file_list)):
        svs_file = svs_directory + svs_file_list[i]
        slide = ops.OpenSlide(svs_file)
        print(svs_file)

        svs_file = svs_directory + svs_file_list[i]
        SVS_to_PNGPatch(svs_file)
