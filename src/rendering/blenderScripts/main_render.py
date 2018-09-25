from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt

import numpy as np
import os

os.chdir('/home/bokoo/Desktop/sft_dl_3dc/data/training_defRenders')
testcld = np.load('testRender0.npy')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(testcld[:,0], testcld[:,1], testcld[:,2])
plt.show()

