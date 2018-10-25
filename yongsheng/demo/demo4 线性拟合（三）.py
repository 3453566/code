import tensorflow as tf
import pandas as pd
import numpy as np

## prepare the original data
# with tf.name_scope('data'):
#     x_data = np.random.rand(100).astype(np.float32)
#     y_data = 0.3*x_data+0.1

data = pd.read_csv('../../data/yongsheng.csv',usecols=[0,1]) #读第1,2列，的数据表
x_data = np.array(data['x'])[0:100]
y_data = np.array(data['y'])[0:100]
print('x_data=',x_data)
print('y_data=',y_data)

##creat parameters
with tf.name_scope('parameters'):
     # with tf.name_scope('weights'):
     #       weight = tf.Variable(tf.random_uniform([1],-1.0,1.0))
     #       tf.summary.histogram('weight',weight)
     # with tf.name_scope('biases'):
     #       bias = tf.Variable(tf.zeros([1]))
     #       tf.summary.histogram('bias',bias)
     with tf.name_scope('a'):
         a = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
         tf.summary.histogram('a', a)
     with tf.name_scope('b'):
         b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
         tf.summary.histogram('b', b)
     with tf.name_scope('c'):
         c=tf.Variable(tf.zeros([1]))
         tf.summary.histogram('c',c)
##get y_prediction
with tf.name_scope('y_prediction'):
     y_prediction = a*x_data*x_data+b*x_data+c
##compute the loss
with tf.name_scope('loss'):
     loss = tf.reduce_mean(tf.square(y_data-y_prediction))
     tf.summary.scalar('loss',loss)
##creat optimizer
optimizer = tf.train.GradientDescentOptimizer(0.5)
#creat train ,minimize the loss
with tf.name_scope('train'):
     train = optimizer.minimize(loss)
#creat init
with tf.name_scope('init'):
     init = tf.global_variables_initializer()
##creat a Session
sess = tf.Session()
#merged
merged = tf.summary.merge_all()
##initialize
writer = tf.summary.FileWriter("../../logs/", sess.graph)
sess.run(init)
## Loop
for step  in  range(101):
    sess.run(train)
    rs=sess.run(merged)
    writer.add_summary(rs, step)