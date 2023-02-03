import cv2
import numpy as np
from skimage.segmentation import slic
from sklearn.cluster import KMeans
from math import sqrt,exp, log, log10
from matplotlib import pyplot as plt
from scipy import integrate


# returns a map with labels as keys and list of corresponding pixel coordinates with that label  
def makeSuperPixelMap(superpixel_labels):
	smap = {}
	for i in range(superpixel_labels.shape[0]):
		for j in range(superpixel_labels.shape[1]):
			key = superpixel_labels[i][j]
			if key==0:
				continue
			value = (i,j)
			if key in smap:
				smap[key].append(value)
			else:
				smap[key]=[value]
	return smap

def findSuperPixelColors(img,smap):
	sp_color_map={}
	keys = smap.keys()

	for key in smap.keys():
		color = np.zeros(3)
		for pixel in smap[key]:
			# print(img[pixel[0]][pixel[1]])
			color[0] += img[pixel[0]][pixel[1]][0]
			color[1] += img[pixel[0]][pixel[1]][1]
			color[2] += img[pixel[0]][pixel[1]][2]
			# print('color is ',color)
		color = color/len(smap[key])

		sp_color_map[key] = np.array(color).astype(int)
	return sp_color_map

def findSuperPixelCoords(smap):
	sp_coord_map={}
	keys = smap.keys()

	for key in smap.keys():
		coord = (0,0)
		for pixel in smap[key]:
			coord = (coord[0] + pixel[0], coord[1]+pixel[1])
		coord = (int(coord[0]/len(smap[key])) , int(coord[1]/len(smap[key])))

		sp_coord_map[key] = coord
	return sp_coord_map

def l2norm(vec):
	return sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)

def convertDictValuestoNumpy(dictValues):
	xsize = len(dictValues)
	ysize = dictValues[0].size
	nparr = np.zeros((xsize,ysize))
	for i in range(xsize):
		for j in range(ysize):
			nparr[i][j] = dictValues[i][j]
	return nparr

def calcContrastCue(labels,unique_labels,cluster_centers):
	# total no of pixels = no of superpixels
	N = len(labels)
	contrast_que = {}
	for i in range(len(unique_labels)):
		kth_label = unique_labels[i]
		kth_mean = cluster_centers[kth_label]
		con_value = 0
		
		for j in range(len(unique_labels)):
			other_label = unique_labels[j]
			other_mean = cluster_centers[other_label]
			n = list(labels).count(other_label)
			if kth_label!=other_label:
				con_value += l2norm(kth_mean-other_mean) * (n/N)

		contrast_que[kth_label] = con_value
	
	return contrast_que

def calcSpatialCue(img,sp_coord_map,labels,unique_labels,cluster_centers):
	spatial_cue = {}
	h,w, channels = img.shape
	center_o = (h/2,w/2)
	d = sqrt(h**2 + w**2)

	for i in range(len(unique_labels)):
		kth_label = unique_labels[i]
		all_pixels = []
		# iterate through the labels, if same then iterate through the pixels
		for j in range(len(labels)):
			if kth_label==labels[j]:
				#j+1
				all_pixels.append(sp_coord_map[j+1])

		n = list(labels).count(kth_label)
		# sigma = np.std(all_pixels)
		sigma = sqrt(d/2)
		c = sigma * sqrt(2*22/7)
		spatial_value=0
		for point in all_pixels:
			val = sqrt((point[0]-center_o[0])**2 + (point[1]-center_o[1])**2) 
			spatial_value += 1/exp(val/c)
		spatial_value = spatial_value * (1/n)

		spatial_cue[kth_label] = spatial_value
	return spatial_cue

# forms images using the cue values given
def formImage(img,cue,labels,smap):
	cue_values = list(cue.values())
	max_value = max(cue_values)
	min_value = min(cue_values)
	cue_values = np.array(cue_values) - min_value
	# max_value = np.max(cue_values)
	cue_values = cue_values/max_value
	cue_values = cue_values *255

	i=0
	for key in cue:
		cue[key] = cue_values[i]
		i+=1
	# print('Normalized cue is')
	# print(cue)
	# print()

	new_img = np.zeros(img.shape)
	for i in range(len(labels)):
		label = labels[i]
		# i+1
		pixels = smap[i+1]

		for pixel in pixels:
			new_img[pixel[0]][pixel[1]] = int(cue[label])
	return new_img

# displays all the plots
def showPlots(img,contrast_image,spatial_image,final_image):

	fig = plt.figure(figsize=(20, 15))
	fig.add_subplot(221)
	plt.title('Original')
	plt.set_cmap('gray')
	plt.imshow(img)

	fig.add_subplot(222)
	plt.title('Contrast Cue')
	plt.set_cmap('gray')
	plt.imshow(contrast_image)

	fig.add_subplot(223)
	plt.title('Spatial Cue')
	plt.set_cmap('gray')
	plt.imshow(spatial_image)

	fig.add_subplot(224)
	plt.title('Complete Effect')
	plt.set_cmap('gray')
	plt.imshow(final_image)

	fig.suptitle('Plots', fontsize=16)
	plt.savefig('Ans2&3.jpg')
	plt.show()


def ostuThresholding(mat):
	MIN_PIXEL_SUM = 10000000000
	OTSU_THRESH = -1
	unique_values = np.unique(mat)

	for i in range(len(unique_values)):
		temp_thresh = unique_values[i]
		pos1 = mat>=temp_thresh
		pos2 = mat<temp_thresh
		
		l1 = np.array([])
		l2 = np.array([])
		
		if pos1.any():
			l1 = np.array(mat[pos1]).ravel()
		if pos2.any():
			l2 = np.array(mat[pos2]).ravel()
		
		val=1000000
		if len(l1)!=0 and len(l2)!=0:
			val = np.var(l1)*l1.shape[0] + np.var(l2)*l2.shape[0]
		elif len(l1)!=0:
			val = np.var(l1)*l1.shape[0]
		else:
			val = np.var(l2)*l2.shape[0]
		# print(val,MIN_PIXEL_SUM)

		if(val<MIN_PIXEL_SUM):
			OTSU_THRESH=temp_thresh
			MIN_PIXEL_SUM =val
	
	return OTSU_THRESH

def normalizeMat(mat):
	min_value = np.min(np.min(mat))
	max_value = np.max(np.max(mat))
	mat = mat - min_value
	mat = mat/max_value
	return mat

# finds phi, seperation measure
def findPhi(mat):
	mat = normalizeMat(mat)

	otsu_thres = ostuThresholding(mat)
	# print('OTSU IS',otsu_thres)

	# color_values = list(sp_color_map.values())
	# np_color_values = np.array(color_values)
	pos1 = mat>=otsu_thres
	pos2 = mat<otsu_thres

	mean_f = 0
	mean_b = 0
	var_f = 0
	var_b = 0

	if pos1.any():
		l1 = np.array(mat[pos1]).ravel()
		mean_f = np.mean(l1)
		var_f = np.var(l1)
	if pos2.any():
		l2 = np.array(mat[pos2]).ravel()
		mean_b = np.mean(l2)
		var_b = np.var(l2)
	var_f+=0.01
	var_b+=0.01

	# print('mean and var is ',mean_f,mean_b,var_f,var_b)
	term1 = (mean_b*var_f - mean_f*var_b)/(var_f - var_b)
	term2 = sqrt(var_f*var_b)/(var_f-var_b) * sqrt( (mean_f - mean_b)**2 - 2*(var_f-var_b)*(log(sqrt(var_b))-log(sqrt(var_f))))
	
	tempz1 = term1 + term2
	tempz2 = term1 - term2

	
	# print('zstar values are coming out to be ',tempz1,tempz2,'\n')
	
	z_star = tempz1
	# max(tempz1,tempz2)


	func_f = lambda x: exp(-1*((x-mean_f)**2)/var_f)/sqrt(var_f*2*22/7)
	func_b = lambda x: exp(-1*((x-mean_b)**2)/var_b)/sqrt(var_b*2*22/7)

	overlap = integrate.quad(func_f, 0, z_star)[0] + integrate.quad(func_b,z_star,1)[0]
	# print('Loss is ',overlap)

	#set no of bins 
	Loss = abs(overlap)
	gamma = 10
	phi = 1/(1+log10(1+gamma*Loss))

	return phi

def calcDis(x1,y1,a,b,c):
	d = abs((a*x1+b*y1+c))/sqrt(a**2 + b**2)
	return d

def performKmeans(sp_color_list):
	clusters_list = np.arange(15)+1
	dist_points_from_cluster_center = []
	for clusters in clusters_list:
		kmeans = KMeans(n_clusters=clusters, random_state=0)
		kmeans.fit(sp_color_list)
		dist_points_from_cluster_center.append(kmeans.inertia_)

	a = dist_points_from_cluster_center[0] - dist_points_from_cluster_center[-1]
	b = clusters_list[-1] - clusters_list[0]
	c1 = clusters_list[0] * dist_points_from_cluster_center[-1]
	c2 = clusters_list[-1] * dist_points_from_cluster_center[0]
	c = c1-c2

	distance_from_line =[]
	for i in range(len(clusters_list)):
		distance_from_line.append(calcDis(clusters_list[i],dist_points_from_cluster_center[i],a,b,c))

	# find index with max value
	index  = distance_from_line.index(max(distance_from_line))
	optimal_cluster = clusters_list[index]

	kmeans = KMeans(n_clusters=optimal_cluster, random_state=0)
	kmeans.fit(sp_color_list)
	labels = kmeans.labels_
	cluster_centers = kmeans.cluster_centers_
	
	return (optimal_cluster,labels,cluster_centers)



if __name__ == '__main__':
	
	print('Hi..starting the code for ques 2 and 3..')
	
	# image_name ='iiitd.jpg'
	image_name = 'q2&3bird.jpg'
	
	img = cv2.imread(image_name)
	# img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	print('Img dimensions are')
	print(img.shape)
	print()

	#*************** Question 2

	n_segments=163
	compactness=20
	convert2lab=True
	segments = slic(img, n_segments=n_segments, start_label=1,convert2lab=convert2lab,compactness=compactness)

	print('A little about segments')
	print('No of segments is ',np.max(np.max(segments)))
	# print(type(segments))
	# print(len(segments[0]))
	# print(segments)
	print()

	smap = makeSuperPixelMap(segments)
	# print('Now lets see what we created')
	# print(smap.keys())
	# print()

	sp_coord_map = findSuperPixelCoords(smap)
	# print('Whats the associated coord for the superpixel')
	# print(sp_coord_map)
	# print()

	sp_color_map = findSuperPixelColors(img,smap)
	# print('Whats the associated color for the superpixel')
	# print(sp_color_map)
	# print()

	# used for clustering
	sp_color_list = list(sp_color_map.values())
	# sp_color_list = convertDictValuestoNumpy(sp_color_list)
	# print(len(sp_color_list))
	# print('converting these values to list now')
	# print(type(sp_color_list))
	# print(sp_color_list)
	# print()

	# clusters= 4 
	# kmeans = KMeans(n_clusters=clusters, random_state=0)
	# kmeans.fit(sp_color_list)
	# labels = kmeans.labels_
	# cluster_centers = kmeans.cluster_centers_
	optimal_cluster_val,labels,cluster_centers = performKmeans(sp_color_list)
	unique_labels = list(set(labels))
	print('Kmeans has finished its job..lets see the report')
	print('Optimal cluster val is ',optimal_cluster_val)
	print('Label associated with each superpixel is ')
	print(labels)
	# print(unique_labels)
	# print(cluster_centers[0])
	print()
	
	contrast_que = calcContrastCue(labels,unique_labels,cluster_centers)
	print('Contrast cue for each cluster (cluster of superpixels) is (normalisation is done later) ')
	print(contrast_que)
	print()

	spatial_cue = calcSpatialCue(img,sp_coord_map,labels,unique_labels,cluster_centers)
	print('Spatial cue for each cluster (cluster of superpixels) is (normalisation is done later) ')
	print(spatial_cue)
	print()

	# ********** Question 3
	img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	contrast_image = formImage(img,contrast_que,labels,smap)
	phi_contrast = findPhi(contrast_image)

	spatial_image = formImage(img,spatial_cue,labels,smap)
	# print('spatial_image is ')
	# print(spatial_image)
	# print()

	phi_spatial = findPhi(spatial_image)

	print('Phi value for Contrast = ',phi_contrast)
	print('Phi value for Spatial = ',phi_spatial)
	print()


	final_image = contrast_image * phi_contrast +  spatial_image * phi_spatial
	max_value = np.max(np.max(final_image))
	min_value = np.min(np.min(final_image))
	final_image = ((final_image - min_value)/max_value)*255
	
	print('Plotting the results now')
	print()
	showPlots(img,contrast_image,spatial_image,final_image)