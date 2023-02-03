import os
import cv2
import sys 
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzzy
from time import time
sys.setrecursionlimit(10**7) 

def readImage(image_name):
    img = cv2.imread(image_name)
    # img = np.array(img).astype(int)
    print('Image shape is ',img.shape)
    norm_img = np.zeros(img.shape)
    final_image = cv2.normalize(img,  norm_img, 0, 255, cv2.NORM_MINMAX)
    rgb_img = img.reshape((final_image.shape[0] * final_image.shape[1], 3))
    return rgb_img

def dfsAlgo(i,j,x,y,z,dx,dy):
    count=1
    for index in range(len(dx)):
        new_i = i + dx[index]
        new_j = j + dy[index]
        if (new_i<fuzzy_image.shape[0] and new_i >=0 and new_j<fuzzy_image.shape[1] and new_j>=0):
            if(x != fuzzy_image[new_i][new_j][0] or y != fuzzy_image[new_i][new_j][1] or z != fuzzy_image[new_i][new_j][2]):
                rgb_tuple = (fuzzy_image[new_i][new_j][0],fuzzy_image[new_i][new_j][1],fuzzy_image[new_i][new_j][2])
                if rgb_tuple in boundary_dict:
                    boundary_dict[rgb_tuple] += 1
                else:
                    boundary_dict[rgb_tuple] = 1
            elif vis[new_i][new_j]==0:
                vis[new_i][new_j] = 1
                count += dfsAlgo(new_i,new_j,x,y,z,dx,dy)
    return count

def changeColorFuzzyCmeans(cluster_members,clusters):
    image = []
    for pix in cluster_members.T:
        image.append(clusters[np.argmax(pix)])
    return image

def showPlots(img,fuzzy_image,segmented_image):
    fig = plt.figure(figsize=(20, 15))
    fig.add_subplot(221)
    plt.title('Original')
    # plt.set_cmap('gray')
    plt.imshow(img)

    fig.add_subplot(222)
    plt.title('Fuzzy Image')
    # plt.set_cmap('gray')
    plt.imshow(fuzzy_image)

    fig.add_subplot(223)
    plt.title('Segmented Image without Stray Pixels')
    # plt.set_cmap('gray')
    plt.imshow(segmented_image)

    fig.suptitle('Plots', fontsize=16)
    plt.savefig('Ans1.jpg')
    plt.show()

def mergeStrayPixels(i,j,majority_surrounding,x,y,z,dx,dy):        
    for index in range(len(dx)):
        new_i = i + dx[index]
        new_j = j + dy[index]
        
        if(new_i<fuzzy_image.shape[0] and new_i >=0 and new_j<fuzzy_image.shape[1] and new_j>=0):
            if(final_image[new_i][new_j][0] == majority_surrounding[0] and final_image[new_i][new_j][1] == majority_surrounding[1] and final_image[new_i][new_j][2] == majority_surrounding[2]):
                continue
            if(x == fuzzy_image[new_i][new_j][0] and y == fuzzy_image[new_i][new_j][1] and z == fuzzy_image[new_i][new_j][2]):
                final_image[new_i][new_j][0] = majority_surrounding[0]
                final_image[new_i][new_j][1] = majority_surrounding[1]
                final_image[new_i][new_j][2] = majority_surrounding[2]
                mergeStrayPixels(new_i,new_j,majority_surrounding,x,y,z,dx,dy)

def getZeroMat(fuzzy_image):
    return np.zeros((fuzzy_image.shape[0],fuzzy_image.shape[1]))

if __name__ == '__main__':
    IMG_NAME = 'q1img.jpeg'
    ISLAND_SIZE = 200
    temp_img = cv2.imread(IMG_NAME) 
    # temp_img = np.array(temp_img).astype(int)  
    fuzzy_image = temp_img
    final_image = temp_img
    vis = getZeroMat(fuzzy_image)
    vis2 = getZeroMat(fuzzy_image)
    boundary_dict = dict()
    cluster = 6
    rgb_img = readImage(IMG_NAME)
    img = np.reshape(rgb_img, (temp_img.shape[0],temp_img.shape[1],3)).astype(np.uint8)
    shape = np.shape(img)
    dx=[-1, -1, 0, 1, 1, 1, 0, -1]
    dy=[0, 1, 1, 1, 0, -1, -1, -1]

    # start_time = time()
    error = 0.005
    maxiter = 1000
    seed = 42
    cluster_returned, u, u0, d, jm, p, fpc = fuzzy.cluster.cmeans(rgb_img.T, cluster, 2, init=None,seed=seed, error=error, maxiter=maxiter)
    
    new_img = changeColorFuzzyCmeans(u,cluster_returned)
    fuzzy_image = np.reshape(new_img,shape).astype(np.uint8)
    
    # print(fuzzy_image.shape)
    final_image = np.reshape(new_img,shape).astype(np.uint8)
    
    cv2.imshow('Fuzzy Image, with stray pixels ', fuzzy_image)
    cv2.waitKey(1000)
    
    vis = getZeroMat(fuzzy_image)
    vis2 = getZeroMat(fuzzy_image)
    
    for i in range(fuzzy_image.shape[0]):
        for j in range(fuzzy_image.shape[1]):
            if(vis[i][j]==0):
                x = final_image[i][j][0]
                y = final_image[i][j][1]
                z = final_image[i][j][2]
                boundary_dict = {(x,y,z):0}
                vis[i][j]=1
                area_of_island = dfsAlgo(i,j,x,y,z,dx,dy)
                if(area_of_island<=ISLAND_SIZE):
                    majority_surrounding = max(boundary_dict,key = boundary_dict.get) 
                    final_image[i][j][0] = majority_surrounding[0]
                    final_image[i][j][1] = majority_surrounding[1]
                    final_image[i][j][2] = majority_surrounding[2]
                    mergeStrayPixels(i,j,majority_surrounding,x,y,z,dx,dy)
    
    showPlots(temp_img,fuzzy_image,final_image)
    # print('Fuzzy time for cluster',cluster,'is',time() - start_time,'seconds')
    # print()
    print('Completed the program..')
    cv2.imshow('Final Output',final_image)
    cv2.waitKey(5000)