import pytesseract
import os
from PIL import Image
import numpy as np
import argparse
import imutils
import glob
import cv2
import re
import pandas as pd
from tqdm import tqdm
import time


# # construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-t", "--template", required=True, help="Path to template image")
# ap.add_argument("-i", "--images", required=True,
#     help="Path to images where template will be matched")
# ap.add_argument("-v", "--visualize",
#     help="Flag indicating whether or not to visualize each iteration")
# args = vars(ap.parse_args())

# load the image image, convert it to grayscale, and detect edges
template = cv2.imread('./input/3_1_1.png')
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
template = cv2.Canny(template, 50, 200)
(tH, tW) = template.shape[:2]
#cv2.imshow("Template", template)

#image extensions to be checked
extn = ["jpg","png"]

#to be extracted
fields = ["doi","doe","passno","dob","name","surname"]
# fields = ["name"]

#dataframe of all the points to form the bounding box for every field
d = {'x1': [163/385, 278/385,284/385,281/385,126/385,128/385], 
     'x2': [245/385, 373/385,380/385,371/385,249/385,200/385],
     'y1': [181/266, 181/266,45/266,111/266,83/266,60/266],
     'y2': [200/266, 205/266,65/266,131/266,99/266,80/266]}

df = pd.DataFrame(data=d)
df.rename(index={0:'doi',1:'doe',2:'passno',3:'dob',4:'name',5:'surname'},inplace=True)

# A dataframe storing all the fields for all the images
info = pd.DataFrame(columns=['surname', 'name', 'passno','dob','doi','doe'])



def get_info(x,y,img,fields,df1):

	for field in fields:
		y1 = int(y*df.loc[field,'y1'])
		y2 = int(y*df.loc[field,'y2'])
		x1 = int(x*df.loc[field,'x1'])
		x2 = int(x*df.loc[field,'x2'])

		# for xs in range(x1-3,x1+3):
		# 	for xe in range(x2-3,x2+3):
		# 		for ys in range(y1-3,y1+3):
		# 			for ye in range(y2-3,y2+3):


		crop_img = img[y1:y2,x1:x2] 
						# crop_img = img[ys:ye,xs:xe] 		
	    
	    #uncomment to see the cropped image for OCR
		# if field == 'doe':
		# cv2.imshow("Cropped image DOI",crop_img)
		# cv2.waitKey(0)

		filename = "{}.png".format(os.getpid())
		cv2.imwrite(filename, crop_img)

		text = pytesseract.image_to_string(Image.open(filename))
		os.remove(filename)
	    
		if field=='doi' or field == 'dob'or field == 'doe':
			text = re.findall(r"\d\d/\d\d/\d\d\d\d",text)
		else:
			text = re.findall(r"\w*",text)
			text = " ".join(text)
			text = text.strip()

		if len(text)>0:
			# if text == 'AISHWARYA':
			# print(field,"  ...   ",text)
								# print((xs/x)*385,'x1      x2',(xe/x)*385,'  ....    ',(ys/y)*266,'y1       y2',(ye/y)*266)

		if len(text)==0:
			text = 'Empty'
		df1[field]=text

	return df1



def imgName(imagePath):
		imgName = imagePath.split('\\')
		imgName = imgName[-1:]
		imgName = "".join(imgName)
		imgName = imgName.split('.')
		imgName = imgName[0]
		folder = "".join(imgName)
		return folder



for i in extn:
    # loop over the images to find the template in
    for imagePath in glob.glob("./input/*_1."+i):
        # load the image, convert it to grayscale, and initialize the
        # bookkeeping variable to keep track of the matched region
        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found = None

        # loop over the scales of the image
        for scale in np.linspace(0.2, 1.0, 20)[::-1]:
            # resize the image according to the scale, and keep track
            # of the ratio of the resizing
            resized = imutils.resize(gray, width = int(gray.shape[1] * scale))
            r = gray.shape[1] / float(resized.shape[1])

            # if the resized image is smaller than the template, then break
            # from the loop
            if resized.shape[0] < tH or resized.shape[1] < tW:
                break

            # detect edges in the resized, grayscale image and apply template
            # matching to find the template in the image
            edged = cv2.Canny(resized, 50, 200)
            result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
            (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

            # check to see if the iteration should be visualized
            # if args.get("visualize", False):
            #     # draw a bounding box around the detected region
            #     clone = np.dstack([edged, edged, edged])
            #     cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),
            #         (maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
            #     cv2.imshow("Visualize", clone)
            #     cv2.waitKey(0)

            # if we have found a new maximum correlation value, then update
            # the bookkeeping variable
            if found is None or maxVal > found[0]:
                found = (maxVal, maxLoc, r)

        # unpack the bookkeeping variable and compute the (x, y) coordinates
        # of the bounding box based on the resized ratio
        (_, maxLoc, r) = found
        (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
        (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))

        # draw a bounding box around the detected result and display the image
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
        
        img = image[startY:endY,startX:endX]

        # cv2.imshow("Image after template matching", img)
        # cv2.waitKey(0)

        df1 = pd.DataFrame(columns=['surname', 'name', 'passno','dob','doi','doe'])

        y,x = img.shape[:2]

        df1 = get_info(x,y,img,fields,df1)
        
        info = info.append(df1,ignore_index=True)

        iName = imgName(imagePath)

        if not os.path.exists('./output/'+iName+''):
        	os.mkdir('./output/'+iName+'')

        export_csv = df1.to_csv (r'./output/'+iName+'/passInfo_'+iName+'.csv', index = None, header=True)
        export_json = df1.to_json (r'./output/'+iName+'/passInfo_'+iName+'.json' )

        


export_csv = info.to_csv (r'TotalpassInfo.csv', index = None, header=True)
export_json = info.to_json(r'TotalpassInfo.json')
print("CSV generated as passInfotest.csv")

