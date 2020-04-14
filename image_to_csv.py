from PIL import Image
from pylab import *
import os
import pandas as pd
import cv2
import numpy as np

np.set_printoptions(threshold=10000)  # 设置打印数量的阈值

def image():
    # 将图片路径存入列表
    files = []

    # 睁眼图片
    path = 'dataset/dataset_B_Eye_Images/OpenEye/'
    for i in os.listdir(path):
        files.append(path + i)
    open_size = len(files)

    # 闭眼图片
    path = 'dataset/dataset_B_Eye_Images/ClosedEye/'
    for i in os.listdir(path):
        files.append(path + i)

    # 压缩图片
    # for file in files:
    #     im = Image.open(file)
    #     im.thumbnail((24, 24))
    #     # print(im.format, im.size, im.mode)
    #     im.save(file, 'JPEG')
    #     print(file)


    # 读取图片并转为数组
    imgs = []
    for i in range(len(files)):
        img = cv2.imread(files[i], cv2.IMREAD_GRAYSCALE)
        img = np.reshape(img, (24 * 24))
        img = list(img)
        imgs.append(img)
        print(i)

    states = []
    count = 0
    for i in range(len(imgs)):
        if i < open_size:
            states.append('open')
            count += 1
        else:
            states.append('close')

    return imgs, states


def write_csv():
    # 将图片数组和状态数组写入csv文件
    imgs, states = image()

    dataframe = pd.DataFrame({'state': states, 'image': imgs})
    dataframe.to_csv("dataset/dataset.csv", index=False, sep=',')


if __name__ == '__main__':
    write_csv()
