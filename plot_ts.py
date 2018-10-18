"""
======================
Plotting a time series
======================

This example shows how you can plot a single time series.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from DataLoad import cvsDataLoad


# Parameters
n_samples, n_features = 100, 48

# Toy dataset
rng = np.random.RandomState(41)
X = rng.randn(n_samples, n_features)

# Plot the first time series
data = cvsDataLoad.loadData()
dataLine = pd.read_csv('../data/nasdaq100_padding.csv', usecols=[0])
plt.plot(dataLine)
plt.show()
