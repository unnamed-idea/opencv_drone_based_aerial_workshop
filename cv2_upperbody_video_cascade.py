"""
import cv2
import numpy as np
from matplotlib import pyplot as plt
img=cv2.imread("a.jpg.jpg",1)



dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
edges = cv2.Canny(dst,150,90)
cv2.imshow('image',edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
import numpy as np
import cv2
cascPath = r'C:\\Users\talha\Downloads\haarcascade_upperbody.xml'
faceCascade = cv2.CascadeClassifier(cascPath)

cap = cv2.VideoCapture('aerial_people.mp4')

#fgbg = cv2.BackgroundSubtractorMOG()
n=0
while(1):
  #  n+=1
    ret, frame = cap.read()
    
    
   ########### 	#frame = cv2.fastNlMeansDenoisingColored(frame,None,50,50,70,21)
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,200,200)
    #fgmask = fgbg.apply(frame)
    #faces = faceCascade.detectMultiScale(
    #gray,
    #scaleFactor=1.09,
    #minNeighbors=4,
   # maxSize=(150,100),
   # flags = cv2.cv.CV_HAAR_SCALE_IMAGE
#)

#print "Found {0} faces!".format(len(faces))

# Draw a rectangle around the faces
    #for (x, y, w, h) in faces:
     #   cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

   # new=(80*edges+frame)/2
    
    cv2.imshow('edges',edges)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()