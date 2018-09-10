# -*- coding: utf-8 -*-
"""
Some useful bpy (Blender) routines/functions
"""
import bpy
import mathutils
import numpy as np

#-------------
# 1 load a 3d obj mesh into blender's environment
#-------------
objFilename = 'harbour'
objFileType = '.obj'
objFileDir = '/home/bokoo/Desktop/sft_dl_3dc/data/3dObjs/'
objFilePath = objFileDir + objFilename + objFileType

importedObj = bpy.ops.import_scene.obj(filepath=objFilePath)
objInContext = bpy.context.selected_objects[0]
print('Imported name: ', objInContext.name)

#-------------
# 2 set viewport shading to 'TEXTURED'
#-------------
#for area in bpy.context.screen.areas: # iterate through areas in current screen
#    if area.type == 'VIEW_3D':
#        for space in area.spaces: # iterate through spaces in current VIEW_3D area
#            if space.type == 'VIEW_3D': # check if space is a 3D view
#                space.viewport_shade = 'TEXTURED' # set the viewport shading to rendered

#---- alternatively -------
area = next(area for area in bpy.context.screen.areas if area.type == 'VIEW_3D')
space = next(space for space in area.spaces if space.type == 'VIEW_3D')
space.viewport_shade = 'TEXTURED'  # set the viewport shading                

#-------------
# 3 point the camera at the loaded object
#-------------
# get a handle on the loaded 3d object, the camera
my3dobj = []
for object in bpy.data.objects:
    if object.name.startswith(objFilename):
        my3dobj = object
    elif object.name.startswith('Camera'):
        mycam = object
        
# test
print('Loaded 3d object is present in the world at : ', my3dobj.location)
print('The render camera is present in the world at : ', mycam.location)

# compute center of mass of the 3d object
my3dobj_com = Vector((0.0, 0.0, 0.0))
nVertices = len(my3dobj.data.vertices)
for vertex in my3dobj.data.vertices:
    my3dobj_com = my3dobj_com + vertex.co
my3dobj_com /= nVertices

print('Center of mass of the loaded 3d oject = ', my3dobj_com)
    

#-------------
# 3 rotate the object in camera frame 
#-------------
# axis-angle
# euler angles

#-------------
# 4 translate the object in camera frame
#-------------
# unit vector from camera's origin to the 3d object's origin
vecCamTo3dObj = (my3dobj.location - mycam.location)
# normalisation 
vecCamTo3dObj = vecCamTo3dObj/np.sqrt(np.sum(np.square(vecCamTo3dObj)))

# translation along the view direction
my3dobj.location += 50 * vecCamTo3dObj 

# position the lamp X units from object's origin along it's x direction
my3dobj_xdir = Vector((1.0, 0.0, 0.0))
my3dobj_xdir_inWorld = my3dobj.matrix_world * my3dobj_xdir
print(my3dobj_xdir)
print(my3dobj_xdir_inWorld)

bpy.data.objects['Lamp'].location = my3dobj.location + 5*my3dobj_xdir_inWorld

#-------------
# 5 apply a modifier to the object
#-------------

#-------------
# 6 save renders
#-------------




