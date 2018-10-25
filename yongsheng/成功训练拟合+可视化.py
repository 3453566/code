#https://blog.csdn.net/csmqq/article/details/51461696
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from keras.layers.advanced_activations import LeakyReLU
from sklearn import preprocessing
#from keras.utils.visualize_plots import figures
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import TensorBoard

import pandas
from DataLoad import cvsDataLoad


# load dataset
x,y = cvsDataLoad().loadData_yongsheng()

fig, ax = plt.subplots()
ax.plot(x, y, 'r')

# part3: create models, with 1hidden layers    
model = Sequential()
model.add(Dense(100, init='uniform', input_dim=1))
# model.add(Activation(LeakyReLU(alpha=0.01)))
model.add(Activation('relu'))

model.add(Dense(50))
# model.add(Activation(LeakyReLU(alpha=0.1)))
model.add(Activation('relu'))

model.add(Dense(1))
# model.add(Activation(LeakyReLU(alpha=0.01)))
model.add(Activation('tanh'))

# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer="rmsprop", metrics=["accuracy"])
# model.compile(loss='mean_squared_error', optimizer=sgd, metrics=["accuracy"])

tbCallBack = TensorBoard(log_dir='../../logs',  # log 目录
                 histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
               #            batch_size=32,     # 用多大量的数据计算直方图
                 write_graph=True,  # 是否存储网络结构图
                 write_grads=True, # 是否可视化梯度直方图
                 write_images=True,# 是否可视化参数
                 embeddings_freq=0,
                 embeddings_layer_names=None,
                 embeddings_metadata=None)

# model.fit(x_train, y_train, nb_epoch=64, batch_size=20, verbose=0)   
hist = model.fit(x, y, batch_size=10, nb_epoch=1, shuffle=True, verbose=0, validation_split=0.2, callbacks=[tbCallBack])
print(hist.history)
score = model.evaluate(x, y, batch_size=10)


out = model.predict(x, batch_size=1)
# plot prediction data

ax.plot(x, out, 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
#figures(hist)

# test
print('\nTesting ------------')
cost = model.evaluate(x, y, batch_size=40)
print('test cost:', cost)
W, b = model.layers[1].get_weights()
print('Weights=', W, '\nbiases=', b)

