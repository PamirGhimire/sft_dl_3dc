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
objFilename = 'horses'
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


print('Placing the camera 50 centimeters from the template ...')
mycam.location = my3dobj.location + 50 * my3dobj.matrix_world * Vector((1.0, 0.0, 0.0))

print('Turning the camera to face the origin of the 3d object')
direction = my3dobj.matrix_world * Vector((-1.0, 0.0, 0.0))
# point the cameras '-Z' and use its 'Y' as up
rot_quat = direction.to_track_quat('-Z', 'Y')
# assume we're using euler rotation
mycam.rotation_euler = rot_quat.to_euler()
bpy.context.scene.update()

#-------------
# 3 rotate the object in camera frame 
#-------------
# axis-angle (world-axis)
# Apply the rotation matrix to the object's world matrix
#my3dobj.matrix_world = mathutils.Matrix.Rotation(np.pi/10.0, 4, Vector((0.0, 1.0, 0.0)))*my3dobj.matrix_world
#bpy.context.scene.update()

# axis-pivot-angle
pivot_world = mycam.matrix_world * Vector((0.0, 0.0, -20.0))
axis_world = Vector((0.0, 0.0, 1.0))
angle = 10.0*(2*np.pi/180.0)
# pivot-axis-angle = subtract pivot, then axis-angle rotation
my3dobj.location -= pivot_world
bpy.context.scene.update()
my3dobj.matrix_world = mathutils.Matrix.Rotation(angle, 4, axis_world) * my3dobj.matrix_world
my3dobj.location += pivot_world
bpy.context.scene.update()


# euler angles

#-------------
# 4 translate the object in camera frame
#-------------
#my3dobj.location += mycam.matrix_world * Vector((10.0, 0.0, 0.0))

# position the lamp X units from object's origin along it's x direction
bpy.data.objects['Lamp'].location = mycam.matrix_world * Vector((0.0, 0.0, 0.0))
bpy.context.scene.update()

#
##-------------
## 5 compute center of mass of the 3d object
##-------------
#my3dobj_com = Vector((0.0, 0.0, 0.0))
#nVertices = len(my3dobj.data.vertices)
#for vertex in my3dobj.data.vertices:
#    my3dobj_com = my3dobj_com + vertex.co
#my3dobj_com /= nVertices
#print('Center of mass of the loaded 3d oject = ', my3dobj_com)


#-------------
# 6 apply a modifier to the object
#-------------
# transform object to a predefined pose in the world space (0, 0, 0, pi/2, 0, 0)
my3dobj.location = Vector((0.0, 0.0, 0.0))
my3dobj.rotation_euler = Vector((np.pi/2.0, 0.0, 0.0)) 



coordinates = [
    ((-1, 0, 0), (-0.7, 0, 0), (-1, 0.5521, 0)),
    ((0, 1, 0), (-0.5521, 1, 0), (0, 0.7, 0))]
#    ,
#    ((0, 0, 0), (0, 0.3, 0), (-0.3, 0, 0))
#]

    
def MakeCurveQuarter(objname, curvename, cList, origin=(0,0,0)):    
    curvedata = bpy.data.curves.new(name=curvename, type='CURVE')    
    curvedata.dimensions = '2D'    
    
    objectdata = bpy.data.objects.new(objname, curvedata)    
    objectdata.location = origin
    
    bpy.context.scene.objects.link(objectdata)    
    
    polyline = curvedata.splines.new('BEZIER')    
    polyline.bezier_points.add(len(cList)-1)    

    for idx, (knot, h1, h2) in enumerate(cList):
        point = polyline.bezier_points[idx]
        point.co = knot
        point.handle_left = h1
        point.handle_right = h2
        point.handle_left_type = 'FREE'
        point.handle_right_type = 'FREE'

    polyline.use_cyclic_u = False 

MakeCurveQuarter("NameOfMyCurveObject", "NameOfMyCurve", coordinates)  

# add a bezier curve to the context : shift+a > curve > Bezier
# Curve Data:
curveData = bpy.data.curves.new('myBCurve', type='CURVE')
curveData.dimensions = '3D'
curveData.resolution_u = 2
# set its 4 coordinates ([top-left, top-right, bottom-left, bottom-right]) 
# when viewdir is s.t. 'y' increases along it (units: centimeters)
coords = [(8.0601, -0.742, 0)]#, (1.858, -19.89, 0.167)]#, (-164.67,-28.59,-0.265), (-37.064, 27.821, 0.2583)] 
# make sure that nothing is selected in the 'path/curve deform' in curve shape
curveData.use_radius = False 

# map coords to spline
polyline = curveData.splines.new('BEZIER')
polyline.bezier_points.add(len(coords))
for i, coord in enumerate(coords):
    x,y,z = coord
    polyline.bezier_points[i].co = (x, y, z)

# create Object
myBCurveObj = bpy.data.objects.new('myBCurve', curveData)

# attach to scene and validate context
scn = bpy.context.scene
scn.objects.link(myBCurveObj)
scn.objects.active = myBCurveObj
myBCurveObj.select = True

# change 'dimension' (in n-panel) of the added bezier curve
#   x = 10 cm, y = 3.7141 cm, z = 0 cm
#myBCurveObj.dimensions = Vector((10.0, 3.7141, 0))
current_x, current_y, current_z =  myBCurveObj.dimensions
myBCurveObj.dimensions = [10.0, 3.7141, current_z]


# change the pose of the curve (0, 0, 0, 0, 90, -90)
myBCurveObj.location = Vector((0.0, 0.0, 0.0))
myBCurveObj.rotation_euler = Vector((0, np.pi/2, -np.pi/2))


# change the 'shape' of the curve from 3D to 2D
# right-click select the curve, tab into edit mode, change curve vertices to...

# select the object (right click on the object)
# add a curve modifier
# select the object (right click on the object)
# add a curve modifier to the object
ob = my3dobj
ob.modifiers.new(name='mysubsurf', type='CURVE')
#ob.modifiers["mysubsurf"].levels = 2
bpy.ops.object.modifier_apply(apply_as='DATA', modifier="mysubsurf")

# select 'BezierCurve' as 'object'
# rotate the curve about x-axis



#-------------
# 7 save renders
#-------------
renderFilePath = '/home/bokoo/Desktop/sft_dl_3dc/data/training_defRenders/testRender.jpg'
bpy.data.scenes["Scene"].render.filepath = renderFilePath
bpy.ops.render.render( write_still=True )





