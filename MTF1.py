import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pyts.image import MTF
from DataLoad import cvsDataLoad

data = cvsDataLoad.loadData()

image_size = 24
mtf = MTF(image_size)
X_mtf = mtf.fit_transform(data)

# Show the results for the first time series
plt.figure(figsize=(8, 8))
plt.imshow(X_mtf[11], cmap='rainbow', origin='lower')

plt.show()






