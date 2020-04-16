import datetime
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers import Input, Activation, Conv2D, Flatten, Dense, MaxPooling2D, merge
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau


plt.style.use('dark_background')


def train():
    # 读取数据集
    x_train = np.load('dataset/x_train.npy').astype(np.float32)
    y_train = np.load('dataset/y_train.npy').astype(np.float32)
    x_val = np.load('dataset/x_val.npy').astype(np.float32)
    y_val = np.load('dataset/y_val.npy').astype(np.float32)

    print(x_train.shape, y_train.shape)
    print(x_val.shape, y_val.shape)

    # 预览
    # plt.subplot(2, 1, 1)
    # plt.title(str(y_train[0]))
    # plt.imshow(x_train[0].reshape((100, 100)), cmap='gray')
    # plt.subplot(2, 1, 2)
    # plt.title(str(y_val[4]))
    # plt.imshow(x_val[4].reshape((100, 100)), cmap='gray')

    # 数据增强
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2
    )
    val_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow(
        x=x_train, y=y_train,
        batch_size=32,
        shuffle=True
    )
    val_generator = val_datagen.flow(
        x=x_val, y=y_val,
        batch_size=32,
        shuffle=False
    )

    # 构建模型

    # Input: 输入层
    inputs = Input(shape=(24, 24, 1))

    # Conv2D: 2D 卷积层
    # filters: 输出空间的维数（即卷积过滤器的数量）
    # kernal_size: 卷积窗的高和宽
    # strides: 横向和纵向的步长
    # padding = 'same': 表示不够卷积核大小的块就补0,所以输出和输入形状相同
    # activation = 'relu': 激活函数, ReLU(Rectified Linear Unit,修正线性单元)

    # 2D输入的最大池化层
    # pool_size: 池窗口的大小

    net = Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu')(inputs)
    net = MaxPooling2D(pool_size=2)(net)

    net = Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu')(net)
    net = MaxPooling2D(pool_size=2)(net)

    net = Conv2D(128, kernel_size=3, strides=1, padding='same', activation='relu')(net)
    net = MaxPooling2D(pool_size=2)(net)

    # 注意力机制AM
    attention_probs = Dense(128, activation='softmax', name='attention_vec')(net)
    attention_mul = merge.multiply([net, attention_probs])

    # 数据压平
    net = Flatten()(attention_mul)

    # Dense: 全连接神经网络层
    net = Dense(512)(net)
    net = Activation('relu')(net)
    net = Dense(1)(net)
    outputs = Activation('sigmoid')(net)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

    model.summary()

    # 训练
    start_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    model.fit_generator(
        train_generator, epochs=50, validation_data=val_generator,
        callbacks=[
            ModelCheckpoint('models/%s.h5' % (start_time), monitor='val_acc', save_best_only=True, mode='max', verbose=1),
            ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=10, verbose=1, mode='auto', min_lr=1e-05)
        ]
    )

    # 精度计算
    from sklearn.metrics import accuracy_score

    model = load_model('models/%s.h5' % start_time)

    y_pred = model.predict(x_val/255.)
    y_pred_logical = (y_pred > 0.5).astype(np.int)

    print('test acc: %s' % accuracy_score(y_val, y_pred_logical))



if __name__ == '__main__':
    train()





