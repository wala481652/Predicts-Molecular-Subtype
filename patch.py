import openslide as ops
import os


def SVS_to_PNGPatch(svs_file):
    # 去除副檔名
    file_basename = os.path.basename(svs_file).split('.')[0]

    # 建立目錄
    if os.path.exists('./database/PNG/total/' + file_basename) == False:
        os.mkdir('./database/PNG/total/' + file_basename)
        print(file_basename+"已建立")
    else:
        print(file_basename+"已存在")

    # 讀取SVS
    slide = ops.OpenSlide(svs_file)

    # 讀取SVS的長寬及Patch的位子
    [w, h] = slide.level_dimensions[0]
    print(w, h)
    [N, M] = [w/512, h/512]
    print(N, M)

    # 儲存Patch為PNG
    for i in range(int(N)):
        for j in range(int(M)):
            region = slide.read_region((i*512, j*512), 0, (512, 512))
            region.save('./database/PNG/total/' + file_basename + '/' +
                        file_basename + '_' + str(i) + '_' + str(j) + '.png')
            print(file_basename + '_' + str(i) + '_' + str(j) + '.png')


if __name__ == '__main__':
    svs_directory = './database/SVS/'
    svs_file_list = os.listdir(svs_directory)

    for k in range(len(svs_file_list)):
        svs_file = svs_directory + svs_file_list[k]
        SVS_to_PNGPatch(svs_file)

    # svs_file = './database/SVS/01.svs'
    # SVS_to_PNGPatch(svs_file)
