# please note, all tutorial code are running under python3.5.
# If you use the version like python2.7, please modify the code accordingly

# 4 - Regressor example

import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import pandas

# load dataset
dataframe = pandas.read_csv("../../data/yongsheng1.csv", header=None)
dataset = dataframe.values
x = dataset[:, 0:3].astype(float).reshape(500,3)
y = dataset[:, 3].astype(float).reshape(500,1)
# plot data
#plt.scatter(x[2], y)
#plt.show()

#X_train, Y_train = x[:160], y[:160]     # first 160 data points
#X_test, Y_test = x[160:], y[160:]       # last 40 data points

# build a neural network from the 1st layer to the last layer
model = Sequential()
model.add(Dense(3, input_shape=(3,)))
model.add(Dense(1))

# choose loss function and optimizing method
model.compile(loss='mse', optimizer='sgd')

# training
print('Training -----------')
for step in range(301):
    cost = model.train_on_batch(x, y)
    if step % 100 == 0:
        print('train cost: ', cost)

# test
print('\nTesting ------------')
cost = model.evaluate(x, y, batch_size=40)
print('test cost:', cost)
W, b = model.layers[0].get_weights()
print('Weights=', W, '\nbiases=', b)

# plotting the prediction
y_pred = model.predict(x)
plt.scatter(x[2], y)
plt.plot(x[2], y_pred)
plt.show()
