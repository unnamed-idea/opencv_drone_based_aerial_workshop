import cv2
import numpy as np 
import color_quantize
def find_frontal_face(img):
	face_cascade = cv2.CascadeClassifier('haarcascade_mcs_upperbody.xml')


	#img=color_quantize.main(img,9)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	faces = face_cascade.detectMultiScale(gray, 1.1, 10)
	for (x,y,w,h) in faces:	
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = img[y:y+h, x:x+w]
	return img

if __name__ == '__main__':
	img=find_frontal_face(cv2.imread('aerial_me.jpg'))    
	cv2.imwrite('img.jpg',img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

