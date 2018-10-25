from keras.models import Sequential
from keras.layers import Dense, Activation
import pandas
import matplotlib.pyplot as plt

# load dataset
dataframe = pandas.read_csv("../../data/yongsheng1.csv", header=None)
dataset = dataframe.values
x = dataset[:, 2].astype(float).reshape(500,1)
y = dataset[:, 3].astype(float).reshape(500,1)

# plot dataset
plt.scatter(x, y)
plt.show()

model = Sequential([
    Dense(32, input_shape=(500,)),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])
"""
model = Sequential()
model.add(Dense(32, input_dim=784))
model.add(Activation('relu'))
"""
# 均方误差回归问题
model.compile(optimizer='rmsprop',
              loss='mse')