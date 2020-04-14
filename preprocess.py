import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import csv


def read_csv(path):
    width = 24
    height = 24
    dims = 1

    # 读取字典格式的csv文件
    file = open(path, 'r')
    reader = csv.DictReader(file)
    rows = list(reader)

    # imgs是一个包含所有图像的numpy数组
    imgs = np.empty((len(list(rows)), height, width, dims), dtype=np.uint8)

    # tgs是一个带有图像标签的numpy数组
    tgs = np.empty((len(list(rows)), 1))

    for row, i in zip(rows, range(len(rows))):
        # 将列表转换回图像格式
        img = row['image']
        img = img.strip('[').strip(']').split(', ')
        im = np.array(img, dtype=np.uint8)
        im = im.reshape((height, width))
        im = np.expand_dims(im, axis=2)
        imgs[i] = im

        # 睁眼的标签为1，闭眼的标签为0
        tag = row['state']
        if tag == 'open':
            tgs[i] = 1
        else:
            tgs[i] = 0

    # 随机打乱数据集
    index = np.random.permutation(imgs.shape[0])
    imgs = imgs[index]
    tgs = tgs[index]

    # 返回图像及其对应的标签
    return imgs, tgs


def preprocess(path):
    # 读取dataset.csv
    X, y = read_csv(path)

    # print(X.shape, y.shape)

    # 预处理

    n_total = len(X)
    X_result = np.empty((n_total, 24, 24, 1))

    for i, x in enumerate(X):
        img = x.reshape((24, 24, 1))

        X_result[i] = img

    from sklearn.model_selection import train_test_split

    x_train, x_val, y_train, y_val = train_test_split(X_result, y, test_size=0.1)

    np.save('dataset/x_train.npy', x_train)  # 划分出的训练集数据
    np.save('dataset/y_train.npy', y_train)  # 划分出的训练集数据
    np.save('dataset/x_val.npy', x_val)  # 划分出的测试集标签
    np.save('dataset/y_val.npy', y_val)  # 划分出的测试集标签


if __name__ == '__main__':
    preprocess('dataset/dataset.csv')









