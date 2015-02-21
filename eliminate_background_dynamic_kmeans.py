import cv2
import numpy as np
from sklearn.cluster import KMeans 
import color_quantize
def draw_rect(img,obj):

	try:
		if obj.shape[1]==2:
			for i in obj:
				x,y,w,h=cv2.boundingRect(obj)
				cv2.rectangle(img, (x,y),(x+w,y+h), (0,0,255), 3)
	except:
		
		x,y,w,h=cv2.boundingRect(obj)
		cv2.rectangle(img, (x,y),(x+w,y+h), (0,0,255), 3)
def adj_hsv(h,tolerance,isup):
	#h=arr[0]
	
	if isup:
		s=v=255
		h*=tolerance
		if h>248:
			h=255
		elif h<1:
			h=10	
	else:
		s=v=50 #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		h*=tolerance
		if h<10:
			h=s=v=0

	h=np.uint8(h)
	
	return [h,s,v]

def find_kmean_colors(img,no_cluster=4):
	img_list= img.reshape((img.shape[0] * img.shape[1], 3))

	clt = KMeans(no_cluster).fit(img_list)

	return clt

def findcontours(img,low_area=3000,high_area=10000):

	contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

	for cnt in contours:
		area = cv2.contourArea(cnt)
		if area > low_area and area<high_area:
			x,y,w,h=cv2.boundingRect(cnt)
			cv2.rectangle(img_original, (x,y),(x+w,y+h), (0,0,255), 3)
			
			#blobs=np.append(blobs,cnt,axis=0)
	#print blobs ,blobs.shape


def finddomcolor(labels,cluster_centers_):
	numLabels = np.arange(0, len(np.unique(labels)) + 1)
	(hist, _) = np.histogram(labels, bins = numLabels)
	
	# normalize the histogram, such that it sums to one
	hist = hist.astype("float")
	hist /= hist.sum()

	cntrs=zip(hist,cluster_centers_)
	minim=0
	for i,j in cntrs:
		if i>minim:
			color=j
			minim=i
	color = np.array(color,dtype=np.uint8).reshape(1,1,3)  # (4,3) array
	
	return color		

def findblobs(img):
	pass

img_original=cv2.imread('aerial_me.jpg')
img=img_original
img,color=color_quantize.main(img,4)
print color
#img=cv2.bilateralFilter(img,9,75,75)

img_hsv =cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
"""
clt=(find_kmean_colors(img,4))

color=finddomcolor(clt.labels_,clt.cluster_centers_)	
		
color_hsv=cv2.cvtColor(color,cv2.COLOR_BGR2HSV)

color_hsv=np.squeeze(color_hsv,axis=1)
"""
upmask=np.uint8(adj_hsv(color[0][0][0],1.2,1))
lowmask=np.uint8(adj_hsv(color[0][0][0],0.6,0))


mask=cv2.inRange(img_hsv,lowmask,upmask)


res=cv2.bitwise_and(img_original,img_original,mask=255-mask)

res_gry=cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)


findcontours(res_gry)

#draw_rect(img_original,blobs)

cv2.imshow('img',img_original)
#cv2.imwrite('output.jpg',img_original)
cv2.waitKey(0)
cv2.destroyAllWindows()
