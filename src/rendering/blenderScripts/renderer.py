#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 16:14:55 2018

@author: Pamir Ghimire
Renderer class
"""
import bpy
import numpy as np


class Renderer:
    # constructor
    def __init__(self):
        print('Renderer object ')
        # camera (contains background image)
        self.mRendCamera = bpy.data.objects['Camera']
        # lamp
        self.mLamp = bpy.data.objects['Lamp']
        # Bezier curve modifier
        self.mBCurveModifier = []
        # 3d object
        self.m3dObj = []
        # radius of imaging sphere
        self.mRadiusCaptureSphere = -1
        # path to 3d mesh
        self.mPathTo3dObj = ''      
        # bezier curve modifier
        self.mCurveObjName = 'bCurveObject'
        self.mCurveName = 'bCurve'
        self.mBCurveObj = []
        self.mBCurveModifierName = 'bCurveModifier'
        #mCamera = Camera()
        self.mRenderWidth = 1900
        self.mRenderHeght = 1080

    # import 3d mesh 
    def f3dObjImport(self, path2Obj):
        self.mPathTo3dObj = path2Obj
        bpy.ops.import_scene.obj(filepath=self.mPathTo3dObj)
        
        obj3d = path2Obj.split('.')[0].split('/')[-1]
        self.m3dObj = bpy.data.objects[obj3d]
        objInContext = bpy.context.selected_objects[0]
        print('Imported name: ', objInContext.name)
        area = next(area for area in bpy.context.screen.areas if area.type == 'VIEW_3D')
        space = next(space for space in area.spaces if space.type == 'VIEW_3D')
        space.viewport_shade = 'TEXTURED'  # set the viewport shading 
        bpy.context.scene.update()
        
    # set location of 3d obj. in the world
    def f3dObjSetLocation_world(self, worldLocation):
        self.m3dObj.location = worldLocation
        bpy.context.scene.update()
            
    # set location of 3d obj. in the world
    def f3dObjSetRotation_world(self, worldEulerAngle):
        self.m3dObj.rotation_euler = worldEulerAngle
        bpy.context.scene.update()
    
    # get centroid of the 3d obj member
    def f3dObjGetCentroid(self):
        my3dobj_com = Vector((0.0, 0.0, 0.0))
        nVertices = len(self.m3dObj.data.vertices)
        for vertex in self.m3dObj.data.vertices:
            my3dobj_com = my3dobj_com + vertex.co
        my3dobj_com /= nVertices
        print('Center of mass of the loaded 3d oject = ', my3dobj_com)
        return my3dobj_com
        
    # get min. x, y and z of the 3d obj member
    def f3dObjGetMins(self):
        minx = float('inf')
        miny = float('inf')
        minz = float('inf')
        for vertex in self.m3dObj.data.vertices:
            if vertex[0] < minx:
                minx = vertex[0]
            if vertex[1] < miny:
                miny = vertex[1]
            if vertex[2] < minz:
                minz = vertex[2]
            
        mins = Vector((minx, miny, minz))
        return mins
    
    # get max. x, y and z of the 3d obj member
    def f3dObjGetMaxs(self):
        maxx = -float('inf')
        maxy = -float('inf')
        maxz = -float('inf')
        for vertex in self.m3dObj.data.vertices:
            if vertex[0] > maxx:
                maxx = vertex[0]
            if vertex[1] > maxy:
                maxy = vertex[1]
            if vertex[2] > maxz:
                maxz = vertex[2]
            
        maxs = Vector((maxx, maxy, maxz))
        return maxs
    
    # set a bezier curve modifier for the 3d mesh by specifying control points
    def fBezierCurveSet(self, cList, origin=(0,0,0)):  
        curvedata = bpy.data.curves.new(name=self.mCurveName, type='CURVE')    
        curvedata.dimensions = '2D'    
        curvedata.use_radius = False
        
        objectdata = bpy.data.objects.new(self.mCurveObjName, curvedata)    
        objectdata.location = origin
        
        bpy.context.scene.objects.link(objectdata)    
        
        polyline = curvedata.splines.new('BEZIER')    
        polyline.bezier_points.add(len(cList)-1)    
    
        for idx, (knot, h1, h2) in enumerate(cList):
            point = polyline.bezier_points[idx]
            point.co = knot
            point.handle_left = h1
            point.handle_right = h2
            point.handle_left_type = 'ALIGNED'
            point.handle_right_type = 'ALIGNED'
    
        polyline.use_cyclic_u = False 
        self.mBCurveObj = bpy.data.objects[self.mCurveObjName]
        # set pose of the added curve
        self.mBCurveObj.location = self.m3dObj.location
        self.mBCurveObj.rotation_euler = Vector((0, np.pi/2, -np.pi/2))
        # add the bCurve as a curve modifier to the 3d obj
        self.m3dObj.modifiers.new(name=self.mBCurveModifierName, type='CURVE')
        self.m3dObj.modifiers[self.mBCurveModifierName].object = self.mBCurveObj
        # move the 3d object a little bit along the curve's major axis
        self.m3dObj.location = self.m3dObj.location + Vector((0, 0, -5))
        bpy.context.scene.update()

    # add to rotation of the bcurve modifier about the world x axis    
    def fBezierCurveRotateDeltaX(self, degrees):
        self.m3dObj.modifiers[self.mBCurveModifierName].object.rotation_euler.y +=  degrees * np.pi/180.0
        bpy.context.scene.update()
         
    # set rotation of the bcurve modifier about the world x axis    
    def fBezierCurveSetRotationX(self, degrees):
        self.m3dObj.modifiers[self.mBCurveModifierName].object.rotation_euler.y =  degrees * np.pi/180.0
        bpy.context.scene.update()
        
    # set position of the 3d obj along the major axis of the b curve modifier
    def f3dObjPositionAlongCurve(self, positionZ = -5.0):
        self.m3dObj.location = self.m3dObj.modifiers[self.mBCurveModifierName].object.location + Vector((0, 0, positionZ))
        bpy.context.scene.update()
        
    # randomly position the 3d obj along the major axis of the b curve modifier
    def f3dObjRandPositionAlongCurve(self, minz=-8.0, maxz=3.0):
        rangez = maxz - minz
        randz = np.random.rand() * rangez + minz
        self.m3dObj.location = self.m3dObj.modifiers[self.mBCurveModifierName].object.location + Vector((0, 0, randz))
        bpy.context.scene.update()    
        
    # create a render using the render cam
    def fRendCamRender(self, pathToRender, width, height):
        bpy.data.scenes['Scene'].render.filepath = pathToRender
        bpy.data.scenes['Scene'].render.resolution_x = self.mRenderWidth
        bpy.data.scenes['Scene'].render.resolution_x = self.mRenderHeght
        bpy.ops.render.render( write_still=True )
        
    # change render camera's location in 3d object frame
    def fRenderCamSetLocation3dObj(self, locationInCam):
        locationInWorld = self.m3dObj.matrix_world * locationInCam
        self.mRendCamera.location = locationInWorld
        bpy.context.scene.update()
        
    # change render camera's look direction in 3d object frame
    def fRenderCamSetLookVector(self, lookVector):
        direction = self.m3dObj.matrix_world * lookVector
        # point the cameras '-Z' and use its 'Y' as up
        rot_quat = direction.to_track_quat('-Z', 'Y')
        # assume we're using euler rotation
        self.mRendCamera.rotation_euler = rot_quat.to_euler()
        bpy.context.scene.update()

#----------------------------------------------------------------------------
# imports
import bpy
import mathutils
import numpy as np

# create a renderer object
myrend = Renderer()

# setup lights in the environment : currently in blender startup file
# camera intrinsics : 35 mm focal length, 32 mm sensor size, image resolution

# import 3d object
objFilePath = '/home/ghimire/Desktop/sft_dl_3dc/data/3dObjs/horses.obj'
myrend.f3dObjImport(objFilePath)

# place it at a pre-determined pose
myrend.f3dObjSetLocation_world((0, 0, 0))
#myrend.fSet3dObjRotation_world((0, 0, 0))

# add a bezier curve modifier to the scene with pre-chosen configurations
x= 5
coordinates = [
    ((-x, 0, 0), (-(x+5), -5, 0), ( (x-5), 5, 0)),
    ((x, 0, 0), (0, 0, 0), ((x+5), 0, 0))]

# add a curve modifier to the object
myrend.fBezierCurveSet(coordinates)

# for n. of desired renders 
    # change rotation of the curve modifier
    # move the camera to a random pose wrt the object
    # save the render

r = 100000
alpha = (np.random.rand()*360 - 180) * np.pi/180.0
theta = (np.random.rand()*180 - 90) * np.pi/180.0
x = r*np.cos(theta)*cos(alpha)
y = r*np.cos(theta)*np.sin(alpha)
z = r*sin(theta) 

x =np.random.rand()*100
y = np.random.rand()*100
z = np.random.rand()*100

myrend.fRenderCamSetLocation3dObj(Vector((x, y, z)) )
lookVector = myrend.f3dObjGetCentroid() - myrend.mRendCamera.location
myrend.fRenderCamSetLookVector(lookVector)


