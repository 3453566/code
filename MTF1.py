import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyts.image import MTF

data = pd.read_csv('../data/nasdaq100_padding.csv',usecols=[0]) #读第一列，得到 40560行*1列 的数据表
n_samples, n_features = 100, 144 #想转成 507行*80列 的二维数组，注意：507*80==40560
print('data.size=',data.size)

# #-------------将data数据按行填充为二维矩阵--------------------------------------------
# array2 = np.array(data).reshape(n_samples,n_features) #先将data转成维数组，在转成二维数组。注意：此时是按照 “行” 排
# print('type(array2):',type(array2))
# print(array2)
##归一化
#max=np.max(array2)
#min=np.min(array2)
#for i in range(array2.shape[0]):
#    for j in range(array2.shape[1]):
#       array2[i][j] = (array2[i][j]-min)/(max-min)
# #-------------  END ----------------------------------------------------------------

#-------注意，上面的array2是按照“行”排成的二维矩阵，如果想变成按照“列”排成的二维矩阵， 应该使用下面的代码
array2 = np.array(data[0:14400]).reshape(n_features, n_samples) #先将data转成维数组，在转成二维数组。注意：此时是按照 “行” 排
def trans(array2):
    a=[[] for i in array2[0]]
    for i in array2:    # 原矩阵第 i 行
        for j in range(len(i)):
            a[j].append(i[j]) #i[j]为原矩阵第i行第j列数据，将它append到a矩阵第 j 行后面
    return a

array2_T = np.array(trans(array2))
print('type(array2_T）：',type(array2_T))
print(array2_T)
#归一化
max=np.max(array2_T)
min=np.min(array2_T)
for i in range(array2_T.shape[0]):
    for j in range(array2_T.shape[1]):
        array2_T[i][j] = (array2_T[i][j]-min)/(max-min)
print(array2_T)
print(array2_T.shape)
#------------  END --------------------------------------------------------------------

#----------------- 马赛克-------------------
image_size = 24
mtf = MTF(image_size)
X_mtf = mtf.fit_transform(array2_T)

# Show the results for the first time series
plt.figure(figsize=(8, 8))
plt.imshow(X_mtf[0], cmap='rainbow', origin='lower')
plt.show()
#----------------- END-------------------------

