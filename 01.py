import cv2
import matplotlib.pyplot as plt
import os
patch_directory = './database/PNG/01_norm/'
patach_file_list = os.listdir(patch_directory)

for i in range(len(patach_file_list)):
    file_basename = os.path.basename(patach_file_list[i]).split('.')[0]
    patch_norm_file = patch_directory + patach_file_list[i]
