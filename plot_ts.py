"""
======================
Plotting a time series
======================

This example shows how you can plot a single time series.
"""

import numpy as np
import matplotlib.pyplot as plt

from DataLoad import cvsDataLoad

data = cvsDataLoad.loadData()
# Parameters
n_samples, n_features = 100, 48

# Toy dataset
rng = np.random.RandomState(41)
X = rng.randn(n_samples, n_features)

# Plot the first time series
plt.plot(data[1])
plt.show()
