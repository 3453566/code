
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from keras.layers.advanced_activations import LeakyReLU
from sklearn import preprocessing
#from keras.utils.visualize_plots import figures
import matplotlib.pyplot as plt
import numpy as np

import pandas

# load dataset
dataframe = pandas.read_csv("../../data/yongsheng1.csv", header=None)
dataset = dataframe.values
x = dataset[:, 2].astype(float).reshape(500,1)
y = dataset[:, 3].astype(float).reshape(500,1)

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

# model.fit(x_train, y_train, nb_epoch=64, batch_size=20, verbose=0)   
hist = model.fit(x, y, batch_size=10, nb_epoch=100, shuffle=True, verbose=0, validation_split=0.2)
# print(hist.history)
score = model.evaluate(x, y, batch_size=10)


out = model.predict(x, batch_size=1)
# plot prediction data

ax.plot(x, out, 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
#figures(hist)