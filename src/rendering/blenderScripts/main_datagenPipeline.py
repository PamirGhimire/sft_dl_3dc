# -*- coding: utf-8 -*-
"""
Produces a set of renders of a template by deforming it using Blender's 
Bezier Curve modifier, and assigning it random poses in the camera space 

-Pamir Ghimire, Spetember 16, 2018
"""
# blender imports
import bpy
from renderer import Renderer

# create a renderer object
myrend = Renderer()

# setup lights in the environment : currently in blender startup file
# camera intrinsics : 35 mm focal length, 32 mm sensor size, image resolution

# import 3d object
objFilePath = '/home/ghimire/Desktop/sft_dl_3dc/data/3dObjs/horses.obj'
myrend.fImport3dObj(objFilePath)

# place it at a pre-determined pose
# add a bezier curve modifier to the scene with pre-chosen configurations
# add a curve modifier to the object
# set the bezier curve as curve modifier's object
# move the object along the modifier curve, so that it's centered

# for n. of desired renders 
    # change rotation of the curve modifier
    # move the camera to a random pose wrt the object
    # save the render
