import matplotlib.pyplot as plt
import os

os.environ['path'] = r'D:\10960135\VScode\lib\openslide-win64-20171122\bin' +  os.environ['path']
print(os.environ['path'])
import openslide as ops

svs_path = './database/TCGA Molecular Subtype/PRAD.1-ERG/TCGA-CH-5739-01A-01-BS1.a927a903-38a7-47ff-a7c9-723dba17e8a1.svs'

slide = ops.open_slide(svs_path)

[w, h] = slide.level_dimensions[0]
print(w, h)
[N, M] = [w/1024, h/1024]
print(N, M)

i = 47
j = 44

region_0 = slide.read_region((i*1024, j*1024), 0, (1024, 1024))
region_1 = slide.read_region((i*1024, j*1024), 1, (1024, 1024))

print((i, j))
print((i*1024, j*1024))

plt.subplot(1, 2, 1)
plt.imshow(region_0)
plt.subplot(1, 2, 2)
plt.imshow(region_1)
plt.show()
