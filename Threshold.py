import cv2
import numpy as np
import matplotlib.pyplot as plt
from patch import *

# norm('./database/01_12_42.png', K=3)
path_norm = './database/01_12_42_norm.png'
# 讀取圖檔
img1 = cv2.imread(path_norm)
# img2 = cv2.imread('./database/PNG/norm/gleason5/01_18_46.png')

file_basename = os.path.basename(path_norm).split('.')[0]
print(file_basename)
# plt.imshow(img)
# plt.show()

gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# plt.imshow(gray)
# plt.show()

# 畫出直方圖
hist = plt.hist(gray.ravel(), 256, [0, 256])
plt.show()
# plt.savefig('./database/' + file_basename + '_hist.png')

ret1, th1 = cv2.threshold(gray, 100, 255, type=0)
ret2, th2 = cv2.threshold(gray, 150, 255, type=0)
ret3, th3 = cv2.threshold(gray, 50, 255, type=0)

#顯示原圖(灰階)
plt.subplot(2, 2, 1)
plt.title('origin')
plt.imshow(gray, cmap='gray')

plt.subplot(2, 2, 2)
plt.title('th1')
plt.imshow(th1, cmap='gray')

plt.subplot(2, 2, 3)
plt.title('th2')
plt.imshow(th2, cmap='gray')

plt.subplot(2, 2, 4)
plt.title('img')
plt.imshow(th3, cmap='gray')
plt.show()
