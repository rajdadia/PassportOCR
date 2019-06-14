import numpy as np
import argparse
import pytesseract
import cv2
import os
from PIL import Image



img = cv2.imread("3_1.png")

y,x = img.shape[:2]

print(x,y)


y1 = int(y*455/540)
y2 = int(y*468/540)
x1 = int(x*170/398)
x2 = int(x*257/398)

crop_img = img[y1:y2,x1:x2]

filename = "{}.png".format(os.getpid())
cv2.imwrite(filename, crop_img)

text = pytesseract.image_to_string(Image.open(filename))
os.remove(filename)
print(text)

# show the output images
cv2.imshow(text, crop_img)
cv2.waitKey(0)