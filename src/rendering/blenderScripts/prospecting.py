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
objFileDir = '/home/ghimire/Desktop/sft_dl_3dc/data/3dObjs/'
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
# 4 rotate the object in camera frame 
#-------------
# axis-angle (world-axis)
# Apply the rotation matrix to the object's world matrix
#my3dobj.matrix_world = mathutils.Matrix.Rotation(np.pi/10.0, 4, Vector((0.0, 1.0, 0.0)))*my3dobj.matrix_world
#bpy.context.scene.update()

## axis-pivot-angle
#pivot_world = mycam.matrix_world * Vector((0.0, 0.0, -20.0))
#axis_world = Vector((0.0, 0.0, 1.0))
#angle = 10.0*(2*np.pi/180.0)
## pivot-axis-angle = subtract pivot, then axis-angle rotation
#my3dobj.location -= pivot_world
#bpy.context.scene.update()
#my3dobj.matrix_world = mathutils.Matrix.Rotation(angle, 4, axis_world) * my3dobj.matrix_world
#my3dobj.location += pivot_world
#bpy.context.scene.update()
#
#
# euler angles

#-------------
# 5 translate the object in camera frame
#-------------
#my3dobj.location += mycam.matrix_world * Vector((10.0, 0.0, 0.0))

# position the lamp X units from object's origin along it's x direction
bpy.data.objects['Lamp'].location = mycam.matrix_world * Vector((0.0, 0.0, 0.0))
# set lamp's energy
bpy.data.lamps['Lamp'].energy = 1.0
bpy.context.scene.update()

#
##-------------
## 6 compute center of mass of the 3d object
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
bpy.context.scene.update()

## add a bezier curve to the context : shift+a > curve > Bezier
x= 5
coordinates = [
    ((-x, 0, 0), (-(x+5), -5, 0), ( (x-5), 5, 0)),
    ((x, 0, 0), (0, 0, 0), ((x+5), 0, 0))]
    
def MakeCurveQuarter(objname, curvename, cList, origin=(0,0,0)):    
    curvedata = bpy.data.curves.new(name=curvename, type='CURVE')    
    curvedata.dimensions = '2D'    
    curvedata.use_radius = False
    
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
        point.handle_left_type = 'ALIGNED'
        point.handle_right_type = 'ALIGNED'

    polyline.use_cyclic_u = False 

MakeCurveQuarter("myBCurveObject", "myBCurve", coordinates)  
myBCurveObj = bpy.data.objects['myBCurveObject']
myBCurveObj.select = True

# change the pose of the curve (0, 0, 0, 0, 90, -90)
myBCurveObj.location = Vector((0.0, 0.0, 0.0))
myBCurveObj.rotation_euler = Vector((0, np.pi/2, -np.pi/2))


# change the 'shape' of the curve from 3D to 2D
# right-click select the curve, tab into edit mode, change curve vertices to...

# select the object (right click on the object)
# add a curve modifier
my3dobj.modifiers.new(name='myBCurveModifier', type='CURVE')
my3dobj.modifiers['myBCurveModifier'].object = myBCurveObj

# move the 3d object a little bit along the curve's major axis
my3dobj.location = my3dobj.location + Vector((0, 0, -5))
bpy.context.scene.update()

# select 'BezierCurve' as 'object', rotate the curve about x-axis
rotYDeg = 15
my3dobj.modifiers['myBCurveModifier'].object.rotation_euler.y +=  rotYDeg * np.pi/180.0
bpy.context.scene.update()


#-------------
# 7 save renders
#-------------
renderFilePath = '/home/ghimire/Desktop/sft_dl_3dc/data/training_defRenders/testRender.jpg'
bpy.data.scenes['Scene'].render.filepath = renderFilePath
bpy.data.scenes['Scene'].render.resolution_x = 1920
bpy.data.scenes['Scene'].render.resolution_x = 1080
bpy.ops.render.render( write_still=True )





