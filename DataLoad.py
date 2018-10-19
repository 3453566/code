import pandas as pd
import numpy as np

class cvsDataLoad:
    public_var = "this is load cvs data"
    def __init__(self):
        self1 = self


    #  在类里面定义的函数，统称为方法，方法参数自定义，可在方法中实现相关的操作
    #  创建实例方法时，参数必须包括self，即必须有实例化对象才能引用该方法，引用时不需要传递self实参
    def loadData():
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
def main():
    data1 = cvsDataLoad.loadData()
    print (data1)
if __name__ == "__main__":
    main()



