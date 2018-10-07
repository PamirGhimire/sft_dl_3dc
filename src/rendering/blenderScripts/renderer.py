#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 16:14:55 2018

@author: Pamir Ghimire
Renderer class
"""
import bpy
import numpy as np
import mathutils

class Renderer:
    # constructor
    def __init__(self):
        print('Renderer object ')
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
        self.mBCurveP0HandleLeft = [] # all handle points are Vector(())'s
        self.mBCurveP0HandleRight = [] 
        self.mBCurveP1HandleLeft  = []
        self.mBCurveP1HandleRight = []  
        #mCamera = Camera()
        # camera (contains background image)
        self.mRenderCam = bpy.data.objects['Camera']
        self.mRenderWidth = 960
        self.mRenderHeight = 540
        self.mRenderCamFocalLengthmm = 7.6800494
        self.mSensorWidthmm = 7.0
        self.mSensorHeightmm = 18.0
        self.fRenderCamSetRenderWidthHeight(self.mRenderWidth, self.mRenderHeight)
        self.fRenderCamSetFocalLengthmm(self.mRenderCamFocalLengthmm)
        self.fRenderCamSetSensorWidthHeightmm(self.mSensorWidthmm, self.mSensorHeightmm)
        self.mRenderCamCalibration = self.fRenderCamGetCalibrationMatrix()

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
            my3dobj_com = my3dobj_com + self.m3dObj.matrix_world*vertex.co
        my3dobj_com /= nVertices
        print('Center of mass of the loaded 3d oject = ', my3dobj_com)
        return my3dobj_com
        
    # get min. x, y and z of the 3d obj member (in model coordinates)
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
    
    # get max. x, y and z of the 3d obj member (in model coordinates)
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
        # store original configuration of the curve (for modification later)
        self.mBCurveP0HandleLeft = self.mBCurveObj.data.splines[0].bezier_points[0].handle_left
        self.mBCurveP0HandleRight = self.mBCurveObj.data.splines[0].bezier_points[0].handle_right
        self.mBCurveP1HandleLeft = self.mBCurveObj.data.splines[0].bezier_points[1].handle_left
        self.mBCurveP1HandleRight = self.mBCurveObj.data.splines[0].bezier_points[1].handle_right    
        bpy.context.scene.update()

    # add to rotation of the bcurve modifier about the world x axis    
    def fBezierCurveRotateDeltaX(self, degrees):
        self.m3dObj.modifiers[self.mBCurveModifierName].object.rotation_euler.y +=  degrees * np.pi/180.0
        bpy.context.scene.update()
        
    # add random changes to the curvature of the bezier curve
    def fBezierCurveRandDistort(self):
        self.mBCurveObj.data.splines[0].bezier_points[0].handle_right +=  1.9*Vector((0.0, np.random.rand()-0.5, 0.0)) 
        self.mBCurveObj.data.splines[0].bezier_points[1].handle_left += 1.9*Vector((0.0, np.random.rand()-0.5, 0.0))
       
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
    def fRenderCamRender(self, pathToRender):
        bpy.data.scenes['Scene'].render.filepath = pathToRender
        bpy.ops.render.render( write_still=True )
      
    # set width and height of render (this affects the rend. cam intrinsics)
    def fRenderCamSetRenderWidthHeight(self, w, h):
        self.mRenderWidth = w
        self.mRendHeight = h
        bpy.data.scenes['Scene'].render.resolution_x = w
        bpy.data.scenes['Scene'].render.resolution_y = h
        bpy.data.scenes['Scene'].render.resolution_percentage = 100.0
        self.mRenderCamCalibration = self.fRenderCamGetCalibrationMatrix()
        
    # set focal length of the render camera (this affects the cam intrinsics)
    def fRenderCamSetFocalLengthmm(self,f=7.6800494):
        self.mRenderCamFocalLengthmm = f
        bpy.data.cameras['Camera'].lens = f
        self.mRenderCamCalibration = self.fRenderCamGetCalibrationMatrix()
        
    # set width and height of the render camera's sensor
    def fRenderCamSetSensorWidthHeightmm(self, w=7.0, h=18.0):
        bpy.data.cameras['Camera'].sensor_width = w
        bpy.data.cameras['Camera'].sensor_height = h
        self.mSensorWidthmm = w
        self.mSensorHeightmm = h
        self.mRenderCamCalibration = self.fRenderCamGetCalibrationMatrix()
        
    # set focal length of the camera
    def fRenderCamSetFocalLengthmm(self, f=7.680):
        bpy.data.cameras['Camera'].lens = f
        self.mRenderCamCalibration = self.fRenderCamGetCalibrationMatrix()

    # get intrinsic matrix of the rendering camera
    def fRenderCamGetCalibrationMatrix(self):
        camd = bpy.data.cameras['Camera']
        f_in_mm = camd.lens
        scene = bpy.context.scene
        resolution_x_in_px = scene.render.resolution_x
        resolution_y_in_px = scene.render.resolution_y
        scale = scene.render.resolution_percentage / 100
        sensor_width_in_mm = camd.sensor_width
        sensor_height_in_mm = camd.sensor_height
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
        if (camd.sensor_fit == 'VERTICAL'):
            # the sensor height is fixed (sensor fit is horizontal), 
            # the sensor width is effectively changed with the pixel aspect ratio
            s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio 
            s_v = resolution_y_in_px * scale / sensor_height_in_mm
        else: # 'HORIZONTAL' and 'AUTO'
            # the sensor width is fixed (sensor fit is horizontal), 
            # the sensor height is effectively changed with the pixel aspect ratio
            pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
            s_u = resolution_x_in_px * scale / sensor_width_in_mm
            s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm

        # Parameters of intrinsic calibration matrix K
        alpha_u = f_in_mm * s_u
        alpha_v = f_in_mm * s_u # same n. of pixels/mm along x and y
        u_0 = resolution_x_in_px * scale / 2
        v_0 = resolution_y_in_px * scale / 2
        skew = 0 # only use rectangular pixels
        K = mathutils.Matrix(
            ((alpha_u, skew,    u_0),
            (    0  , alpha_v, v_0),
            (    0  , 0,        1 )))

        return K          
        
    # change render camera's location in 3d object frame
    def fRenderCamSetLocation3dObj(self, locationInCam):
        locationInWorld = self.m3dObj.matrix_world * locationInCam
        self.mRenderCam.location = locationInWorld
        bpy.context.scene.update()
        
    # change render camera's look direction in 3d object frame
    def fRenderCamSetLookAtPoint(self, pointInWorld):
        # camera's location in the world
        loc_camera = self.mRenderCam.matrix_world.to_translation()
        direction = pointInWorld - loc_camera
        # point the cameras '-Z' and use its 'Y' as up
        rot_quat = direction.to_track_quat('-Z', 'Y')
        # assume we're using euler rotation
        self.mRenderCam.rotation_euler = rot_quat.to_euler()
        bpy.context.scene.update()
        
    # save mesh vertices in camera frame
    def fRenderCamSaveMeshVertsInCamFrame(self, savename, meshIsTwoFaced=True):
        scene = bpy.context.scene
        obj_data = self.m3dObj.to_mesh(scene, apply_modifiers=True, settings='PREVIEW')
        verts = [v.co for v in obj_data.vertices]

        allVerts = []        
        for vertex in verts:
            matWorldToCam = mathutils.Matrix(self.mRenderCam.matrix_world)
            matWorldToCam.invert()
            
            vert = matWorldToCam * self.m3dObj.matrix_world * vertex
            vert = [vert.x, vert.y, vert.z]
            allVerts.append(vert)
        
        if not meshIsTwoFaced:
            np.save(savename, allVerts)
            return np.array(allVerts)
        else:
            nonDuplicateVerts = []
            for j in range( int(len(allVerts)/2)):
                nonDuplicateVerts.append(allVerts[j])
            np.save(savename, nonDuplicateVerts)
            return np.array(nonDuplicateVerts)
        

    # save mesh vertices in camera frame
    def fRenderCamGetMeshInCamFrame(self, meshIsTwoFaced=True):
        scene = bpy.context.scene
        obj_data = self.m3dObj.to_mesh(scene, apply_modifiers=True, settings='PREVIEW')
        verts = [v.co for v in obj_data.vertices]
        normals = [v.normal for v in obj_data.vertices]

        allVerts = []        
        allNormals = []
        for n in range(len(verts)):
            vertex = verts[n]
            normal = normals[n]
            matWorldToCam = mathutils.Matrix(self.mRenderCam.matrix_world)
            matWorldToCam.invert()
            
            vert = matWorldToCam * self.m3dObj.matrix_world * vertex
            normal = matWorldToCam * self.m3dObj.matrix_world * normal
            vert = [vert.x, vert.y, vert.z]
            normal = [normal.x, normal.y, normal.z]
            
            allVerts.append(vert)
            allNormals.append(normal)
        
        if not meshIsTwoFaced:
            return np.array(allVerts), np.array(allNormals)
        else:
            nonDuplicateVerts = []
            nonDuplicateNormals = []
            for j in range( int(len(allVerts)/2)):
                nonDuplicateVerts.append(allVerts[j])
                nonDuplicateNormals.append(allNormals[j])
            return np.array(nonDuplicateVerts), np.array(nonDuplicateNormals)

#-------------------------------------
# imports
import bpy
import mathutils
import numpy as np

projectDir = '/home/bokoo/Desktop/sft_dl_3dc'
# create a renderer object
myrend = Renderer()

# setup lights in the environment : currently in blender startup file
# camera intrinsics : 7.68 mm focal length, 7mm(w) x 18mm(h) sensor, 960x540 
myrend.fRenderCamSetRenderWidthHeight(w=640, h=480)
myrend.fRenderCamSetSensorWidthHeightmm(w=7.0,h=18.0)
myrend.fRenderCamSetFocalLengthmm(f=7.68)

# import 3d object
objFilePath = projectDir + '/data/3dObjs/horses_frontBack.obj'
myrend.f3dObjImport(objFilePath)

# place it at a pre-determined position
myrend.f3dObjSetLocation_world((0, 0, 0))

# add a bezier curve modifier to the scene with pre-chosen configurations
x= 5
coordinates = [
    ((-x, 0, 0), (-(x+5), -5, 0), ( (x-5), 5, 0)),
    ((x, 0, 0), (0, 0, 0), ((x+5), 0, 0))]

# add a curve modifier to the object
myrend.fBezierCurveSet(coordinates)

# generate renders
nCameraPositions = 1000 # also sets the number of different deformations
nRendersPerCamPosAndDeformation = 5

# mean distance between the 3d object and the camera
rObjCam = 50 #centimeters
# for n. of desired renders
for nCameraPos in range(nCameraPositions): 
    # change rotation of the curve modifier
    myrend.fBezierCurveRotateDeltaX(4.0)
    
    # change curvature controlling params of the curve
    myrend.fBezierCurveRandDistort()
   
    # change the distance of the camera from the object
    randRObjCam = rObjCam + 10*(np.random.rand() - 0.5)
    
    # move the camera to a random pose wrt the object
    alpha = (np.random.rand()*180 - 90) * np.pi/180.0
    theta = (np.random.rand()*180 - 90) * np.pi/180.0
    x = randRObjCam*np.cos(theta)*cos(alpha)
    y = randRObjCam*np.cos(theta)*np.sin(alpha)
    z = randRObjCam*sin(theta) 

    for nRenderPerPosDef in range(nRendersPerCamPosAndDeformation):
        print('Render counter : ', (nCameraPos*nRendersPerCamPosAndDeformation) + nRenderPerPosDef)

        myrend.fRenderCamSetLocation3dObj(Vector((x, y, z)) )
        lookPoint = myrend.f3dObjGetCentroid() + Vector((np.random.rand()*10.0-5.0, np.random.rand()*10.0-5.0, np.random.rand()*10.0-5.0))
        myrend.fRenderCamSetLookAtPoint(lookPoint)
    
        # change the focal length of the camera
        randFocal = 7.0#6.0 + 2 * np.random.rand()
        myrend.fRenderCamSetFocalLengthmm(randFocal)    
        currCalibrationMatrix = myrend.fRenderCamGetCalibrationMatrix()
        
        # save the render
        pathToRender = projectDir + '/data/training_defRenders/testRender' + str(nCameraPos*nRendersPerCamPosAndDeformation + nRenderPerPosDef) +'.jpg'
        myrend.fRenderCamRender(pathToRender)    
    
        # save the cloud, and other data
        pathToSaveRendData = projectDir + '/data/training_defRenders/testRender' + str(nCameraPos*nRendersPerCamPosAndDeformation + nRenderPerPosDef)
        #allVerts = myrend.fRenderCamSaveMeshVertsInCamFrame(savename=pathToSaveCld, meshIsTwoFaced=True)
        allVerts, allNormals = myrend.fRenderCamGetMeshInCamFrame(meshIsTwoFaced=True)
        rendData = dict()
        
        mesh = dict()
        mesh['vertices'] = allVerts
        mesh['normals'] = allNormals
        rendData['mesh'] = mesh
        
        rendData['focalLength'] = randFocal
        rendData['cameraMatrix'] = np.array(currCalibrationMatrix)
        np.save(pathToSaveRendData, rendData)
    
#---------END--OF--RENDERING--PIPELINE---