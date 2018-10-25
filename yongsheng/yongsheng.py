EPOCHS = 30
DEBUG_LEVEL = 1

import keras
import numpy
import pandas
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load dataset
dataframe = pandas.read_csv("../../data/yongsheng.csv", header=None)
dataset = dataframe.values
X = dataset[:, 0:3].astype(float)
Y = dataset[:, 3].astype(float)

# create model
model = Sequential()
model.add(Dense(8, input_dim=3,  name="layer1"))
model.add(Dense(1, activation='relu', name="layer2"))
# Compile model
model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])

# fit & log
tensorBoardCallBack = keras.callbacks.TensorBoard(log_dir='../../logs',
                                                  histogram_freq=0,
                                                  write_graph=True,  # 是否存储网络结构图
                                                  write_grads=True,  # 是否可视化梯度直方图
                                                  write_images=True)

model.fit(X, Y, epochs=EPOCHS, batch_size=5, verbose=DEBUG_LEVEL, callbacks=[tensorBoardCallBack],
          validation_split=0.66)

