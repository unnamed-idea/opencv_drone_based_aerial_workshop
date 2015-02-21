from sklearn.cluster import KMeans 
def find_kmean_colors(img,no_cluster=2,uplow_tolerance=1.1):
	clt = KMeans(no_cluster).fit(img)
	

	up_hsv=list(uplow_tolerance*clt.cluster_centers_)
	low_hsv=list((2-uplow_tolerance)*clt.cluster_centers_)

	return up_hsv,low_hsv