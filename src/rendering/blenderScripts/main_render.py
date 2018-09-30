projectDir = '/home/bokoo/Desktop/sft_dl_3dc'
import sys
sys.path.insert(0, projectDir+'/src/utils/')

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt

import numpy as np
import cv2
import os

os.chdir(projectDir + '/data/training_defRenders')
plt.close("all")



# sanity check : project 3d points onto the image 
n = 0
im = cv2.imread('testRender' + str(n) + '.png')
imc = im.copy()
vertsNormals = np.load('testRender' + str(n) + '.npy').item()
cloud = vertsNormals['vertices']
k = np.load('calibration.npy')

imwidth = im.shape[1]
imheight = im.shape[0]


for i in range(10):
    np.load('testRender' + str(i) + '.npy').item()
    cldToLoad = 'testRender' + str(i) + '.npy'
    cloud = vertsNormals['vertices']    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(testcld[:,0], testcld[:,1], testcld[:,2])
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()

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

##---------------------------------
#import ObjLoader
## check texture info
#pathToObjFile = projectDir + '/data/3dObjs/horses.obj'
#pathToTexture = projectDir + '/data/textures/horses.jpg'
#
#verts, texs, normals = ObjLoader.loadObj(pathToObjFile)
#textureMap = cv2.imread(pathToTexture)
#
#canvas = np.zeros(imc.shape)
#canvas_im = np.zeros(imc.shape)
#for nPoint in range(cloud.shape[0]):
#    pCam = cloud[nPoint,:] 
#    
#    pIm = k.dot(pCam)
#    pIm = pIm / (5.0*pIm[2])
#    
#    if (pIm[0] >= 0 and pIm[0] < imwidth): # valid x coordinate
#        if (pIm[1] >= 0 and pIm[1] < imheight): # valid y coordinate
#            row = int(pIm[1])
#            col = int(pIm[0])
#            
#            uv = texs[nPoint]
#            uv0 = int(uv[0] * textureMap.shape[1])-1 # width, x
#            uv1 = int(uv[1] * textureMap.shape[0])-1 # height, y
#            canvas[row, imwidth-col] = textureMap[uv1, uv0] # (row, col)
#            canvas_im[row, imwidth-col] = im[row, imwidth-col]
#            
#plt.figure()
#plt.imshow(canvas)
#plt.figure()
#plt.imshow(canvas_im)