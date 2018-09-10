import cv2
import os

resizeWidth = 595
resizeHeight = 842	

allfiles = os.listdir()
for somefile in allfiles:
	if (somefile.endswith('jpg') or somefile.endswith('jpeg')) and not somefile.startswith('r_'):
		print('Resizing ', somefile,  ' ...')
		# read image		
		im = cv2.imread(somefile)
		
		# get width (n. cols) and height (n. rows) of the image		
		width = im.shape[1]
		height = im.shape[0]

		# if width < height, width = 600, height = width/0.707
		if (width < height):
			width_r = resizeWidth
			height_r= resizeHeight

		# else, height = 600, width = height/0.707
		else:
			height_r = resizeWidth
			width_r = resizeHeight

		# resize the image to (width_r, height_r)
		im_r = cv2.resize(im, (width_r, height_r))

		# write to disk with a modified filename
		cv2.imwrite("r_" + somefile, im_r)
