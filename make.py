import pandas as pd               # 資料處理套件
import matplotlib.pyplot as plt   # 資料視覺化套件
import numpy as np
import csv
file = '1100414.csv'

exampleFile = open('./database/CSV/'+file)  # 打开csv文件
exampleReader = csv.reader(exampleFile)  # 读取csv文件
exampleData = list(exampleReader)  # csv数据转换为列表
length_zu = len(exampleData)  # 得到数据行数
length_yuan = len(exampleData[0])  # 得到每行长度

# for i in range(1,length_zu):
#     print(exampleData[i])

x = list()
y = list()
z = list()
j = list()
k = list()

for i in range(1, length_zu):  # 从第二行开始读取
    x.append(float(exampleData[i][0]))  # 将第一列数据从第二行读取到最后一行赋给列表x
    y.append(float(exampleData[i][1]))  # 将第二列数据从第二行读取到最后一行赋给列表
    z.append(float(exampleData[i][2]))  # 将第二列数据从第二行读取到最后一行赋给列表
    j.append(float(exampleData[i][4]))
    k.append(float(exampleData[i][5]))

if ".csv" in file:
    while "." in file:
        file = file[:-1]

p1, = plt.plot(x, y, '-')
p2, = plt.plot(x, z, '-')
plt.title(file)
plt.xlabel("epoch")
plt.legend([p2, p1], ["loss", "accuracy"])
plt.ylim(0, 1)
plt.savefig('./chart/'+file+'-1.png')
plt.show()
p1, = plt.plot(x, j, '-')
p2, = plt.plot(x, k, '-')
plt.title(file)
plt.xlabel("epoch")
plt.legend([p2, p1], ["val_loss", "val_accuracy"])
plt.ylim(0, 1)
plt.savefig('./chart/'+file+'-2.png')
plt.show()
