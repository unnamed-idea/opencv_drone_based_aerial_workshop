from sklearn.cluster import MiniBatchKMeans
import numpy as np
import argparse
import cv2
import time
def main(image,no_cluster=4):
	# load the image and grab its width and height
	#image = cv2.imread(str(image))
	
	image=cv2.bilateralFilter(image,9,75,75)
	
	(h, w) = image.shape[:2]

	# convert the image from the RGB color space to the L*a*b*
	# color space -- since we will be clustering using k-means
	# which is based on the euclidean distance, we'll use the
	# L*a*b* color space where the euclidean distance implies
	# perceptual meaning
	image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

	# reshape the image into a feature vector so that k-means
	# can be applied
	image = image.reshape((image.shape[0] * image.shape[1], 3))

	# apply k-means using the specified number of clusters and
	# then create the quantized image based on the predictions
	clt = MiniBatchKMeans(no_cluster)
	
	labels = clt.fit_predict(image)

	quant = clt.cluster_centers_.astype("uint8")[labels]

	# reshape the feature vectors to images
	quant = quant.reshape((h, w, 3))
	

	# convert from L*a*b* to RGB
	quant = cv2.cvtColor(quant, cv2.COLOR_HSV2BGR)

	numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
	(hist, _) = np.histogram(clt.labels_, bins = numLabels)
	
	# normalize the histogram, such that it sums to one
	hist = hist.astype("float")
	hist /= hist.sum()

	cntrs=zip(hist,clt.cluster_centers_)
	minim=0
	for i,j in cntrs:
		if i>minim:
			color=j
			minim=i
	color = np.array(color,dtype=np.uint8).reshape(1,1,3)  # (4,3) array
	
	
	return quant,color

if __name__ == '__main__':
	img=cv2.imread('aerial_me.jpg')
	output,_=main(img,4)
	cv2.imshow('img',output)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
