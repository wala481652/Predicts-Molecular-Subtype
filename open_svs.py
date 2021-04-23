import openslide as ops
import matplotlib.pyplot as plt

svs_path = './database/SVS/05.svs'

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
