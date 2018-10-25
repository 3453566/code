import pandas as pd


import matplotlib.pyplot as plt
import numpy as np

import pandas

from sklearn import preprocessing

class cvsDataLoad:
    public_var = "this is load cvs data"
    def __init__(self):
        self1 = self


    #  在类里面定义的函数，统称为方法，方法参数自定义，可在方法中实现相关的操作
    #  创建实例方法时，参数必须包括self，即必须有实例化对象才能引用该方法，引用时不需要传递self实参
    def loadData(self):
        data = pd.read_csv('../data/nasdaq100_padding.csv', usecols=[2])  # 读第一列，得到 40560行*1列 的数据表
        n_samples, n_features = 338, 120  # 想转成 507行*80列 的二维数组，注意：507*80==40560
        array2 = np.array(data[0:40560]).reshape(n_features, n_samples)  # 先将data转成维数组，在转成二维数组。注意：此时是按照 “行” 排

        array2_T = np.array(cvsDataLoad.trans(array2))
        # 归一化
        max = np.max(array2_T)
        min = np.min(array2_T)
        for i in range(array2_T.shape[0]):
            for j in range(array2_T.shape[1]):
                array2_T[i][j] = (array2_T[i][j] - min) / (max - min)
        return array2_T

    def trans(array2):
        a = [[] for i in array2[0]]
        for i in array2:  # 原矩阵第 i 行
            for j in range(len(i)):
                a[j].append(i[j])  # i[j]为原矩阵第i行第j列数据，将它append到a矩阵第 j 行后面
        return a

    def loadData_yongsheng(self):
        # load dataset
        dataframe = pandas.read_csv("../../data/yongsheng1.csv", header=None)
        dataset = dataframe.values
        x = dataset[:, 2].astype(float).reshape(len(dataframe), 1)
        y = dataset[:, 3].astype(float).reshape(len(dataframe), 1)

        x = preprocessing.scale(x)
        scaler = preprocessing.StandardScaler().fit(x)
        y = scaler.transform(y)

        # 例子：将数据缩放至[0, 1]间。训练过程: fit_transform()
        min_max_scaler = preprocessing.MinMaxScaler()
        y = min_max_scaler.fit_transform(y)
        # 将上述得到的scale参数应用至测试数据
        y_test_minmax = min_max_scaler.transform(y)  # out: array([[-1.5 ,  0. , 1.66666667]])
        # 可以用以下方法查看scaler的属性
        min_max_scaler.scale_  # out: array([ 0.5 ,  0.5,  0.33...])
        min_max_scaler.min_
        return x,y

def main():
    x,y = cvsDataLoad.loadData_yongsheng()
    print (x,y)
if __name__ == "__main__":
    main()



