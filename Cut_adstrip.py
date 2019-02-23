import cv2 
import numpy as np
from PIL import Image

#import images
path = "./ADstrip.png"
orignal = cv2.imread(path)
mask = cv2.imread(path) #for Createing mask image


#detect edges
canny_edges = cv2.Canny(orignal, 120, 250 )


#detect contours
dst, contours, hierarchy = cv2.findContours(canny_edges,
	 cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


#get size of origial image
height, width, channel = orignal.shape
img_size = height*width


# black out AD-strips
targets = []

for i, contour in enumerate(contours):
	area = cv2.contourArea(contour)
	if img_size*0.99 < area:
		continue
	targets.append(contour)
	#cv2.drawContours( orignal, contour, -1, ( 0, 255, 0), 3)

targets = np.array(targets)
cv2.fillPoly( mask, targets, [0, 0, 0] )

#create mask image of AD-strips
gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
ret, dst = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
mask2= cv2.bitwise_not(dst)

#convert images from opencv to Pillow(PIL)
orignal = orignal[:,:,::-1] #from BGR to RGB
orignal = Image.fromarray(orignal)
mask2 = Image.fromarray(mask2)

# cut background
bg = Image.new("RGBA", orignal.size, (0, 0, 0, 0))
bg.paste(orignal, (0,0), mask2.split()[0])

#output image
bg.show()

