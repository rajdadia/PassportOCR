import os
import pytesseract
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
from flask import Flask, flash, request, redirect, url_for,send_from_directory
from werkzeug.utils import secure_filename



# load the image image, convert it to grayscale, and detect edges
template = cv2.imread('./template/3_1.png')
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
template = cv2.Canny(template, 50, 200)
(tH, tW) = template.shape[:2]
#cv2.imshow("Template", template)

#image extensions to be checked
extn = ["jpg","png"]

#to be extracted
fields = ["doi","doe","passno","dob","name","surname"]


#dataframe of all the points to form the bounding box for every field
d = {'x1': [170/398, 293/398,298/398,299/398,140/398,138/398], 
     'x2': [258/398, 389/398,390/398,382/398,281/398,270/398],
     'y1': [455/540, 453/540,316/540,379/540,357/540,333/540],
     'y2': [471/540, 480/540,338/540,400/540,371/540,349/540]}

df = pd.DataFrame(data=d)
df.rename(index={0:'doi',1:'doe',2:'passno',3:'dob',4:'name',5:'surname'},inplace=True)


def get_info(x,y,img,fields,df1):

    for field in fields:
        y1 = int(y*df.loc[field,'y1'])
        y2 = int(y*df.loc[field,'y2'])
        x1 = int(x*df.loc[field,'x1'])
        x2 = int(x*df.loc[field,'x2'])

        crop_img = img[y1:y2,x1:x2] 
        
        #uncomment to see the cropped image for OCR
            #cv2.imshow("Cropped image DOI",crop_img)

        filename = "{}.png".format(os.getpid())
        cv2.imwrite(filename, crop_img)

        text = pytesseract.image_to_string(Image.open(filename))
        os.remove(filename)
        #cv2.waitKey(0)
        if field=='doi' or field == 'dob'or field == 'doe':
            text = re.findall(r"\d\d/\d\d/\d\d\d\d",text)
        else:
            text = re.findall(r"\w*",text)
            text = " ".join(text)
            text = text.strip()

        df1[field]=text

    return df1



def imgName(imagePath):
        imgName = imagePath.split('.')
        imgName = imgName[0]
        folder = "".join(imgName)
        return folder


UPLOAD_FOLDER = './uploads/'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_files',
                                    filename=filename))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

@app.route('/uploads/<filename>')
def uploaded_files(filename):
	
    # A dataframe storing all the fields for all the images
    info = pd.DataFrame(columns=['surname', 'name', 'passno','dob','doi','doe'])

    # for i in extn:
    # loop over the images to find the template in
    for imagePath in glob.glob("./uploads/"+filename):
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

        #cv2.imshow("Image after template matching", img)

        df1 = pd.DataFrame(columns=['surname', 'name', 'passno','dob','doi','doe'])

        y,x = img.shape[:2]

        df1 = get_info(x,y,img,fields,df1)
        
        info = info.append(df1,ignore_index=True)

        iName = imgName(filename)

        if not os.path.exists('./output/'+iName+''):
            os.mkdir('./output/'+iName+'')

        export_csv = df1.to_csv (r'./output/'+iName+'/passInfo_'+iName+'.csv', index = None, header=True)
        print(df1)
        export_json = df1.to_json (r'./uploads/'+iName+'.json' )
        filename = iName+".json"
        print("\nIname is ...."+iName+"\nfilename is ...."+filename+"\n")


    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


if __name__ == '__main__':
   app.run(host='0.0.0.0')