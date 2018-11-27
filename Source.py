#!/usr/bin/env python
# coding: utf-8

# In[41]:


import numpy as np
import cv2
import imageio
from PIL import Image
import matplotlib.pyplot as plt
import time
import math
import glob
import matplotlib.colors as colors
from skimage import color, io, exposure
from scipy.ndimage import morphology as morph
from skimage.morphology import disk
from skimage.transform import resize
from skimage import filters
#%matplotlib inline


# In[45]:


def calcErrorSurface(panorama, curr_img, overlap, channel):
    A = panorama[:, -overlap-1:, channel]
    B = curr_img[:, 0:overlap+1, channel]
    return np.square(A-B)

def calcSeam(e):
    E = np.zeros(e.shape);
    E[0, :] = e[0, :];
    for h in range(1, e.shape[0]):
        for w in range(0, e.shape[1]):
            if w == 0:
                cost = min(E[h-1, w], E[h-1, w+1]);
            elif w == e.shape[1]-1:
                cost = min(E[h-1, w-1], E[h-1, w]);
            else:
                cost = min(E[h-1, w-1], E[h-1, w], E[h-1, w+1]);
            E[h, w] = e[h, w] + cost;
    return E

def calcSeamPath(E, e):
    h = e.shape[0];
    path = np.zeros((h, 1));
    idx = np.argmin(E[h-1, :]);
    path[h-1] = idx;
    for h in range(e.shape[0]-2,-1,-1):
        w = int(path[h+1][0]);
        if w > 0 and E[h, w-1] == E[h+1, w]-e[h+1, w]:
            path[h] = w-1;
        elif w < e.shape[1] - 1 and E[h, w+1] == E[h+1, w]-e[h+1, w]:
            path[h] = w+1;
        else:
            path[h] = w;

    path[path==0] = 1
    return path
    
def stitchImage(panorama, curr_img, path, overlap):
    n = 1
    bound_threshold = 15;
    
    tmp = np.zeros((0,panorama.shape[1] + curr_img.shape[1] - overlap,3)).astype('float64');
    for h in range(0, panorama.shape[0]):
        A = np.expand_dims(panorama[h, 0:-(overlap-int(path[h][0])+1), :], axis=0);
        B = np.expand_dims(curr_img[h, int(path[h][0])-1:, :], axis = 0);
        ZA = np.concatenate((np.expand_dims(panorama[h,:,:],axis=0), np.zeros((A.shape[0],panorama.shape[1] + curr_img.shape[1] - overlap-np.expand_dims(panorama[h,:,:],axis=0).shape[1],3))), axis=1);
        ZB = np.concatenate((np.expand_dims(panorama[h,0:panorama.shape[1] + curr_img.shape[1] - overlap-np.expand_dims(curr_img[h,:,:],axis=0).shape[1],:], axis=0), np.expand_dims(curr_img[h,:,:],axis=0)), axis=1);
        filt_A = np.ones((1, A.shape[1]-bound_threshold));
        grad = np.expand_dims(np.linspace(1, 0, 2*bound_threshold+1, endpoint=True), axis = 0);
        filt_B = np.zeros((1, B.shape[1]-bound_threshold));
        blender = np.concatenate((filt_A, grad, filt_B), axis=1);
        Z = (blender[:, 0:ZA.shape[1]].T*ZA.T).T + ((1-blender[:, 0:ZB.shape[1]]).T*ZB.T).T;
        tmp = np.concatenate((tmp,Z));
    return tmp

def colorCorrection(images_temp, shift, bestIndex, gamma=2.2):
    alpha = np.ones((3, len(images_temp)));
    for rightBorder in range(bestIndex+1, len(images_temp)):
        for i in range(bestIndex+1, rightBorder+1):
            I = images_temp[i];
            J = images_temp[i-1];
            overlap = I.shape[1] - shift[i-1];
            for channel in range(3):
                alpha[channel, i] = np.sum(np.power(J[:,-overlap-1:,channel], gamma))/np.sum(np.power(I[:,0:overlap+1,channel],gamma));

        G = np.sum(alpha, 1)/np.sum(np.square(alpha), 1);
        
        for i in range(bestIndex+1, rightBorder+1):
            for channel in range(3):
                images_temp[i][:,:,channel] = np.power(G[channel] * alpha[channel, i], 1.0/gamma) * images_temp[i][:,:,channel];
                
    for leftBorder in range(bestIndex-1, -1, -1):
        for i in range(bestIndex-1, leftBorder-1, -1):
            I = images_temp[i];
            J = images_temp[i+1];
            overlap = I.shape[1] - shift[i-1];
            for channel in range(3):
                alpha[channel, i] = np.sum(np.power(J[:,0:overlap+1,channel], gamma))/np.sum(np.power(I[:,-overlap-1:,channel],gamma));

        G = np.sum(alpha, 1)/np.sum(np.square(alpha), 1);
        
        for i in range(bestIndex-1, leftBorder-1, -1):
            for channel in range(3):
                images_temp[i][:,:,channel] = np.power(G[channel] * alpha[channel, i], 1.0/gamma) * images_temp[i][:,:,channel];
    return images_temp

def getBestIndex(images_temp):
    idx = 0
    bestVar = 255**5
    for i in range(len(images_temp)):
        curMeans = np.array([np.mean(images_temp[i][:,:,0]),np.mean(images_temp[i][:,:,1]),np.mean(images_temp[i][:,:,2])]);
#         if -np.var(images_temp[i].flatten()) < bestVar:
        if np.max(curMeans) - np.min(curMeans) < bestVar:
            idx = i
            bestVar = np.max(curMeans) - np.min(curMeans)
#             bestVar = -np.var(images_temp[i].flatten())
    return idx

def calcPanorama(images_dir, shift):
    start = time.time()
    # read panorama source images
    files = glob.glob(images_dir + 'in-*.*g');
    files = sorted(files)
    print(len(files))
    
    image_files = [np.array(Image.open(files[i])) for i in range(len(files))];
    
    images_temp = [ image_files[i].astype('float64') for i in range(len(image_files))];
    
    if image_files[0].ndim == 2 or image_files[0].shape[2] == 1:
        images_temp = [ cv2.resize(cv2.cvtColor(image_files[i], cv2.COLOR_GRAY2RGB), (200, 300)).astype('float64') for i in range(len(image_files))];
    
    bestIndex = getBestIndex(images_temp);
    
    print("The image chosen as the base image for color is the image with index " + str(bestIndex)+'.')
    
    images_temp = colorCorrection(images_temp, shift, bestIndex);
    panorama = images_temp[0];
    for i in range(1, len(images_temp)):
        curr_img = images_temp[i];
        
        channel = np.argmax([np.var(curr_img[:,:,0]), np.var(curr_img[:,:,1]), np.var(curr_img[:,:,2])]);
        
        overlap = curr_img.shape[1] - shift[i-1];
        e = calcErrorSurface(panorama, curr_img, overlap, channel);
        E = calcSeam(e)
        path = calcSeamPath(E,e)
        panorama = stitchImage(panorama, curr_img, path, overlap)
        print("The time taken for merging " + str(i+1) + " images: " + str(time.time() - start))
#     fig = plt.figure(figsize=(20,10))
#     plt.axis('off')
#     plt.imshow(panorama/np.max(panorama));
    print("The image has been saved as output.png")
    imageio.imwrite(images_dir+'output.png', np.array(255*panorama/np.max(panorama)).astype('uint8'));
    return panorama


# In[46]:


calcPanorama('./results/3/', [55]*11);


# In[35]:


calcPanorama('./results/2/', [109]*6);


# In[36]:


calcPanorama('./results/1/', [36]*16);


# In[37]:


calcPanorama('./results/4/', [85]*5);

