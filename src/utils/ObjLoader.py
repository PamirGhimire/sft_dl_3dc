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
