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


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--template", required=True, help="Path to template image")
ap.add_argument("-i", "--images", required=True,
    help="Path to images where template will be matched")
ap.add_argument("-v", "--visualize",
    help="Flag indicating whether or not to visualize each iteration")
args = vars(ap.parse_args())

# load the image image, convert it to grayscale, and detect edges
template = cv2.imread(args["template"])
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
template = cv2.Canny(template, 50, 200)
(tH, tW) = template.shape[:2]

#cv2.imshow("Template", template)
strs = ["jpg","png"]



d = {'x1': [171/398, 293/398,298/398,299/398,140/398,138/398], 
     'x2': [257/398, 389/398,390/398,382/398,281/398,270/398],
     'y1': [455/540, 453/540,316/540,379/540,357/540,333/540],
     'y2': [470/540, 480/540,338/540,400/540,371/540,349/540]}

df = pd.DataFrame(data=d)
df.rename(index={0:'doi',1:'doe',2:'passno',3:'dob',4:'name',5:'surname'},inplace=True)


info = pd.DataFrame(columns=['surname', 'name', 'passno','dob','doi','doe'])

def get_doi(x,y,img):

    y1 = int(y*df.loc['doi','y1'])
    y2 = int(y*df.loc['doi','y2'])
    x1 = int(x*df.loc['doi','x1'])
    x2 = int(x*df.loc['doi','x2'])

    # if y>1000:  
    #     y2 = int(y*473/540)

    crop_img = img[y1:y2,x1:x2] 

    #uncomment to see the cropped image for OCR
    #cv2.imshow("Cropped image DOI",crop_img)

    filename = "{}.png".format(os.getpid())
    cv2.imwrite(filename, crop_img)

    text = pytesseract.image_to_string(Image.open(filename))
    os.remove(filename)
    #cv2.waitKey(0)
    text = re.findall(r"\d\d/\d\d/\d\d\d\d",text)
    return text

def get_doe(x,y,img):

    y1 = int(y*df.loc['doe','y1'])
    y2 = int(y*df.loc['doe','y2'])
    x1 = int(x*df.loc['doe','x1'])
    x2 = int(x*df.loc['doe','x2'])





    crop_img = img[y1:y2,x1:x2] 

    #uncomment to see the cropped image for OCR
    # cv2.imshow("Cropped image Expiry",crop_img)

    filename = "{}.png".format(os.getpid())
    cv2.imwrite(filename, crop_img)

    text = pytesseract.image_to_string(Image.open(filename))
    os.remove(filename)
    #cv2.waitKey(0)
    text = re.findall(r"\d\d/\d\d/\d\d\d\d",text)
    return text

def get_passno(x,y,img):

    y1 = int(y*df.loc['passno','y1'])
    y2 = int(y*df.loc['passno','y2'])
    x1 = int(x*df.loc['passno','x1'])
    x2 = int(x*df.loc['passno','x2'])

    crop_img = img[y1:y2,x1:x2] 

    #uncomment to see the cropped image for OCR
    # cv2.imshow("Cropped image passno",crop_img)

    filename = "{}.png".format(os.getpid())
    cv2.imwrite(filename, crop_img)

    text = pytesseract.image_to_string(Image.open(filename))
    os.remove(filename)
    # cv2.waitKey(0)
    return text

def get_surname(x,y,img):

    y1 = int(y*df.loc['surname','y1'])
    y2 = int(y*df.loc['surname','y2'])
    x1 = int(x*df.loc['surname','x1'])
    x2 = int(x*df.loc['surname','x2'])

    crop_img = img[y1:y2,x1:x2] 

    #uncomment to see the cropped image for OCR
    # cv2.imshow("Cropped image Surname",crop_img)

    filename = "{}.png".format(os.getpid())
    cv2.imwrite(filename, crop_img)

    text = pytesseract.image_to_string(Image.open(filename))
    os.remove(filename)
    #cv2.waitKey(0)
    text = re.findall(r"\w*",text)
    text = " ".join(text)
    text = text.strip()
    return text

def get_name(x,y,img):

    y1 = int(y*df.loc['name','y1'])
    y2 = int(y*df.loc['name','y2'])
    x1 = int(x*df.loc['name','x1'])
    x2 = int(x*df.loc['name','x2'])

    crop_img = img[y1:y2,x1:x2] 

    #uncomment to see the cropped image for OCR
    # cv2.imshow("Cropped image Name",crop_img)

    filename = "{}.png".format(os.getpid())
    cv2.imwrite(filename, crop_img)

    text = pytesseract.image_to_string(Image.open(filename))
    os.remove(filename)
    #cv2.waitKey(0)
    text = re.findall(r"\w*",text)
    text = " ".join(text)
    text = text.strip()
    return text


def get_dob(x,y,img):

    y1 = int(y*df.loc['dob','y1'])
    y2 = int(y*df.loc['dob','y2'])
    x1 = int(x*df.loc['dob','x1'])
    x2 = int(x*df.loc['dob','x2'])

    crop_img = img[y1:y2,x1:x2] 

    #uncomment to see the cropped image for OCR
    # cv2.imshow("Cropped image DOB",crop_img)

    filename = "{}.png".format(os.getpid())
    cv2.imwrite(filename, crop_img)

    text = pytesseract.image_to_string(Image.open(filename))
    os.remove(filename)
    #cv2.waitKey(0)
    text = re.findall(r"\d\d/\d\d/\d\d\d\d",text)
    return text

for i in strs:
    # loop over the images to find the template in
    for imagePath in glob.glob(args["images"] + "/*_1."+i):
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
            if args.get("visualize", False):
                # draw a bounding box around the detected region
                clone = np.dstack([edged, edged, edged])
                cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),
                    (maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
                cv2.imshow("Visualize", clone)
                cv2.waitKey(0)

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

        #cv2.imshow("Image after template", img)

        y,x = img.shape[:2]

        doi = get_doi(x,y,img)

        dob = get_dob(x,y,img)

        surname = get_surname(x,y,img)

        name = get_name(x,y,img)

        passno= get_passno(x,y,img)

        doe = get_doe(x,y,img)

        df1 = pd.DataFrame([[surname,name,passno,dob,doi,doe]], columns=['surname', 'name', 'passno','dob','doi','doe'])

        info = info.append(df1,ignore_index=True)

export_csv = info.to_csv (r'passInfo.csv', index = None, header=True)
print(info.loc[:,'doi'])
print("CSV generated as passInfotest.csv")

