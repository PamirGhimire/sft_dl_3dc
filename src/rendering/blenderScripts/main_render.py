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



# sanity check : project 3d points (vertices of deformed meshes) onto the image 
n = 5
im = cv2.imread('testRender' + str(n) + '.png')
imc = im.copy()
rendData = np.load('testRender' + str(n) + '.npy').item()
cloud = rendData['mesh']['vertices'] #vertsNormals['vertices']
k = rendData['cameraMatrix'] #np.load('calibration.npy')

imwidth = im.shape[1]
imheight = im.shape[0]

# 3d scatter plot of the point cloud
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(cloud[:,0], cloud[:,1], cloud[:,2])
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_title('vertices of the deformed mesh')
plt.show()

projections = []
for nPoint in range(cloud.shape[0]):
    pCam = cloud[nPoint,:] 
    
    pIm = k.dot(pCam)
    pIm = pIm / pIm[2]
    
    if (pIm[0] >= 0 and pIm[0] < imwidth):
        if (pIm[1] >= 0 and pIm[1] < imheight):
            row = int(pIm[1])
            col = imwidth - int(pIm[0])
            
            imc[row, col] = [255, 0, 0]

plt.figure()
plt.title('vertices of deformed mesh projected on image')
plt.imshow(imc)

##---------------------------------
# TEXTURE LOOKUP TESTING:
##---------------------------------
import ObjLoader
def rgb2hex(rgb):
    hexColors = []
    for i in range(rgb.shape[0]):
        r = rgb[i, 0]
        g = rgb[i, 1]
        b = rgb[i, 2]
        hexCol =  "#{:02x}{:02x}{:02x}".format(r,g,b)
        hexColors.append(hexCol)
    return hexColors

def hex2rgb(hexcode):
    return tuple(map(ord,hexcode[1:].decode('hex')))

# check texture info
pathToObjFile = projectDir + '/data/3dObjs/horses.obj'
pathToTexture = projectDir + '/data/textures/horses.jpg'

verts, texs, normals = ObjLoader.loadObj(pathToObjFile)
textureMap = cv2.imread(pathToTexture)
texMapWidth = textureMap.shape[1]
texMapHeight = textureMap.shape[0]

visibleVertices = []
visibleVerticesColors = []
visibleVerticesTrueColors = []
visibleVerticesIdxs = []
idxCounter = 0

for nPoint in range(cloud.shape[0]):
    pCam = cloud[nPoint,:] 
    
    pIm = k.dot(pCam)
    pIm = pIm / (pIm[2])
    
    if (pIm[0] >= 0 and pIm[0] < imwidth): # valid x coordinate
        if (pIm[1] >= 0 and pIm[1] < imheight): # valid y coordinate
            uv = texs[nPoint]
            uv0 = int(uv[0] * textureMap.shape[1])-1 # width, x
            uv1 = int(uv[1] * textureMap.shape[0])-1 # height, y
            visibleVertexTrueColor = textureMap[uv1, uv0] # (row, col)
            
            row = int(pIm[1]) # y
            col = imwidth - int(pIm[0]) # x
            visibleVertexColor = im[row, col]

            visibleVertex = cloud[idxCounter,:]
            
            visibleVerticesIdxs.append(idxCounter)
            visibleVertices.append(visibleVertex)
            visibleVerticesColors.append(visibleVertexColor)
            visibleVerticesTrueColors.append(visibleVertexTrueColor)
            
    idxCounter += 1

visibleVertices = np.array(visibleVertices)
visibleVerticesTrueColors = np.array(visibleVerticesTrueColors)
visibleVerticesColors = np.array(visibleVerticesColors)

visibleVerticesTrueColorsHex = rgb2hex(visibleVerticesTrueColors)
visibleVerticesColorsHex = rgb2hex(visibleVerticesColors)

# visible vertices with true colors (from texture map)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(visibleVertices[:,0], visibleVertices[:,1], visibleVertices[:,2], c=visibleVerticesTrueColorsHex)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_title('true colors of visible vertices from texture map')
plt.show()

# visible vertices with colors taken from render
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(visibleVertices[:,0], visibleVertices[:,1], visibleVertices[:,2], c=visibleVerticesColorsHex)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_title('rendered colors of the visible vertices')
plt.show()
