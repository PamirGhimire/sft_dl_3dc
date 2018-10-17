#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 10:07:38 2018
A simple obj file reader to retrieve vertices, normals, and per-vertex uv's
@author: Pamir Ghimire
"""
    
def loadObj(pathToObjFile):
    with open(pathToObjFile) as f:
        allLines = f.readlines()
        # remove whitespace characters like `\n` at the end of each line
        allLines = [x.strip() for x in allLines]
        vertices = []
        normals = []
        textureCoords = []
        
        for line in allLines:
            # if vertex
            if line.startswith('v '):
                l = line.split(' ')
                v = []
                for i in range(1, len(l)):
                    v.append(float(l[i]))
                vertices.append(v)

            # if texture
            if line.startswith('vt '):
                l = line.split(' ')
                v = []
                for i in range(1, len(l)):
                    v.append(float(l[i]))
                textureCoords.append(v)

            # if normal
            if line.startswith('vn '):
                l = line.split(' ')
                v = []
                for i in range(1, len(l)):
                    v.append(float(l[i]))
                normals.append(v)
                
        # if only a single normal
        if (len(normals) < len(vertices)):
            for i in range(len(vertices) - len(normals)):
                normals.append(normals[0])
                
    return vertices, textureCoords, normals


pathToObjFile = '../../data/3dObjs/horses.obj'
vertices, texcoords, normals = loadObj(pathToObjFile)
#-----------
import numpy as np

# convert vertices list to a numpy array
vertices = np.array(vertices)

# mesh's width is along z (larger dim), height is along y (smaller dim.)
uniqueZs = np.unique(vertices[:,2])
nUniqueZs = len(uniqueZs)

idx = np.argsort(vertices, axis=0)
vertices_sortedZ = vertices[idx[:,2],:]

verticesByZ = []
counter = 0
for i in range(nUniqueZs):
    zverts = []    
    for j in range(vertices_sortedZ.shape[0]):
        if (vertices_sortedZ[j, 2] == uniqueZs[i]):
            zverts.append(vertices_sortedZ[j,:])
        if (vertices_sortedZ[j, 2] > uniqueZs[i]):
            break
    verticesByZ.append(zverts)
    print(str(counter), 'n. vertices for z = ', str(uniqueZs[i]), ' : ', str(len(zverts)) )
    counter += 1

#-----------------------
# size of the grid to which the mesh is to be mapped
width = 65 #cols, decreasing z
height = 33 #rows, decreasing y
texcoords = np.array(texcoords)

gridVertIdxs = -1 * np.ones((height, width))
takenMap = np.zeros(vertices.shape[0])
takenWithDistance = -1*np.ones(vertices.shape[0])

for row in range(height):
    for col in range(width):
        print('(row, col) = (', str(row), ', ', str(col), ')')
        uInv = col/(1.0*width) # col
        vInv = (height - row)/(1.0*height) # row

        uvInv = np.array([uInv, vInv])
        
        # find the uv coordinate closest to uvInv
        dstMin = 1e9        
        for i in range(max(texcoords.shape)): #assuming more than 3 vertices
            dst = np.sum(np.power(uvInv - texcoords[i,:], 2))
            if dst < dstMin:
                if takenMap[i] == 0:
                    dstMin = dst
                    gridVertIdxs[row, col] = int(i)
        
        takenMap[int(gridVertIdxs[row, col])] = 1.0
        takenWithDistance[int(gridVertIdxs[row, col])] = dstMin
        print('idx = ', str(gridVertIdxs[row, col]))

# check
gridVertIdxsUnstacked = np.reshape(gridVertIdxs, gridVertIdxs.shape[0]*gridVertIdxs.shape[1])
uniqueIdxs = np.unique(gridVertIdxsUnstacked)
print('n. of unique indexes = ', str(len(uniqueIdxs)))
print('must be one greater than n. of vertices in the mesh')
print('n. of vertices in the mesh = ', str(vertices.shape[0]) )
print('n. of grids not associated with any mesh vertex = ', str(np.sum(gridVertIdxsUnstacked == -1)) )

for row in range(gridVertIdxs.shape[0]):
    for col in range(gridVertIdxs.shape[1]):
        if gridVertIdxs[row, col] == -1:
            print('row, col: ', str(row), ', ', str(col))

np.save('gridWithVertexIdxs.npy', gridVertIdxs)                     
