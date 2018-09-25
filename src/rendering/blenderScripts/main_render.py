from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt

import numpy as np
import cv2
import os

os.chdir('/home/bokoo/Desktop/sft_dl_3dc/data/training_defRenders')
plt.close("all")

#for i in range(10):
#    cldToLoad = 'testRender' + str(i) + '.npy'
#    testcld = np.load(cldToLoad)
#    
#    fig = plt.figure()
#    ax = fig.add_subplot(111, projection='3d')
#    ax.scatter(testcld[:,0], testcld[:,1], testcld[:,2])
#    ax.set_xlabel('X Label')
#    ax.set_ylabel('Y Label')
#    ax.set_zlabel('Z Label')
#    plt.show()

# sanity check : project 3d points onto the image 
n = 8
im = cv2.imread('testRender' + str(n) + '.png')
imc = im
cloud = np.load('testRender' + str(n) + '.npy')
k = np.load('calibration.npy')
k[1, 1] = k[0, 0]

imwidth = im.shape[1]
imheight = im.shape[0]

projections = []
for nPoint in range(cloud.shape[0]):
    pCam = cloud[nPoint,:] 
    
    pIm = k.dot(pCam)
    pIm = pIm / pIm[2]
    
    if (pIm[0] >= 0 and pIm[0] < imwidth):
        if (pIm[1] >= 0 and pIm[1] < imheight):
            row = int(pIm[1])
            col = int(pIm[0])
            imc[row, imwidth-col] = [255, 0, 0]

plt.figure()
plt.imshow(imc)
    
