#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Loads all files in folder, performs gaussian normalisation based on mask, crops slices that do not include brain mask (e.g. no WHM in the neck)
Saves examples images for each subject. Concatenates each slice so that two files are generated:
    
'images_preprocessed.npy' -> of dimensions nslices,X,Y,2 e.g. 1300,200,176,2 -> 1300 axial slices, 200x176 xy, 2 channels (FLAIR and T1)
'masks_preprocessed.npy' -> of dimensions nslices,X,Y,2 e.g. 1300,200,176,1 -> 1300 axial slices, 200x176 xy, 1 channels (prediction mask)
    
@author: Marcello Venzi
"""
import copy
import os
import argparse
import nibabel as nb
import numpy as np
import matplotlib.pyplot as plt

#%%

#Data specific inputs

#crop images to 200 voxels in X and 176 in Y direction -> plot images to see if that's correct

rows_standard = 200  #the input size 
cols_standard = 176 

#suffix of each file generate by the script preproc_FLAIR_MPRAGE

FLAIR_sub='.FLAIR.N4.nii.gz'
mask_sub='.FLAIR.mask.nii.gz'
T1_sub='.MPRAGE.N4.toFLAIR.nii.gz'

#subfix for each FLAIR mask (done by radiologist)

labels_sub='.wmh.init.mask_bleddyn.nii.gz'

#%%


def arg_parser():
    parser = argparse.ArgumentParser(description='Make a two npy datasets from preprocessed pairs of FLAIR and T1 images')
    required = parser.add_argument_group('Required')
    required.add_argument('-f', '--folder', type=str, required=True,
                        help='folder with all the images (output from preprocessing script)')  
    return parser

#%%

def preprocessing(FLAIR_array, T1_array, label_array, mask_array):
    
    FLAIR_array -=np.mean(FLAIR_array[mask_array == 1])      #Gaussian normalization
    FLAIR_array /=np.std(FLAIR_array[mask_array == 1])
    
    rows_o = np.shape(FLAIR_array)[1]
    cols_o = np.shape(FLAIR_array)[2]
    FLAIR_array = FLAIR_array[:, int((rows_o-rows_standard)/2):int((rows_o-rows_standard)/2)+rows_standard, int((cols_o-cols_standard)/2):int((cols_o-cols_standard)/2)+cols_standard]

    T1_array -=np.mean(T1_array[mask_array == 1])      #Gaussian normalisation
    T1_array /=np.std(T1_array[mask_array == 1])
    T1_array = T1_array[:, int((rows_o-rows_standard)/2):int((rows_o-rows_standard)/2)+rows_standard, int((cols_o-cols_standard)/2):int((cols_o-cols_standard)/2)+cols_standard]
    
    #remove slices that do not contain brain mask
    
    zsum=np.sum(mask_array[:,:,:],axis=0).sum(axis=0) #sum over xy for each z -> zslices
    
    slices = np.reshape(np.nonzero(zsum),-1)
    
    T1_array=T1_array[slices,:,:]
    FLAIR_array=FLAIR_array[slices,:,:]
    mask_array=mask_array[slices,:,:]
    label_array=label_array[slices,:,:]
    
    imgs_two_channels = np.concatenate((FLAIR_array[..., np.newaxis], T1_array[..., np.newaxis]), axis = 3)
    
    label_array = label_array[:, int((rows_o-rows_standard)/2):int((rows_o-rows_standard)/2)+rows_standard, int((cols_o-cols_standard)/2):int((cols_o-cols_standard)/2)+cols_standard]
    
    mask_array = mask_array[:, int((rows_o-rows_standard)/2):int((rows_o-rows_standard)/2)+rows_standard, int((cols_o-cols_standard)/2):int((cols_o-cols_standard)/2)+cols_standard]
    
    return(imgs_two_channels,label_array[...,np.newaxis],mask_array)


#%%

def savefig(imgs_two_channels,label_array,mask_array,pid):
    
    #Select slices with highest intensity (more WMH)
    
    FLAIR_array=np.squeeze(imgs_two_channels[:,:,:,0])
    T1_array=np.squeeze(imgs_two_channels[:,:,:,1])
    label_array=np.squeeze(label_array[:,:,:,0])
    
    zslice=int(np.argmax(np.sum(label_array[:,:,:],axis=1).sum(axis=1))) #sum over xy for each z -> zslices

    my_cmap = copy.copy(plt.cm.get_cmap('jet')) 
    my_cmap.set_bad(alpha=0) # set NaN to zero alpha
    #Save axial slice with highest detection
    pred_mask = np.array(label_array, dtype=float)
    pred_mask[pred_mask < 1] = np.nan
    
    #Plot
    f = plt.figure()
    f.add_subplot(1,2, 1)
    plt.imshow(FLAIR_array[zslice,:,:], cmap='gray', vmin=-1, vmax=1)
    plt.imshow(pred_mask[zslice,:,:], cmap=my_cmap, alpha=1,vmin=0, vmax=1)
    plt.axis('off')
    f.add_subplot(1,2, 2)
    plt.imshow(T1_array[zslice,:,:], cmap='gray', vmin=-1, vmax=1)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    
    plt.savefig(pid+".example_axial_slice_"+str(zslice) +".png",bbox_inches='tight',pad_inches=0.0)    
    
    return


#%%
    
#select all files nii from a folder, save two npy files of dimension nslices(n training examples) x 200 x 200 x 2 (data) and nslices x 200 x 200 x 1 (labels)



def main(args=None):

    args = arg_parser().parse_args(args)
    
    path=args.folder
    os.chdir(path)

    files = []
    for file in os.listdir(path):
        if file.endswith(".nii.gz"):
            files.append(file)

    #list of files in folder with unique labels

    res = [sub.replace('.FLAIR.mask.nii.gz', '').replace('.MPRAGE.N4.toFLAIR.nii.gz', '').replace('.FLAIR.N4.nii.gz','' ).replace('.wmh.init.mask_bleddyn.nii.gz','') for sub in files]
    
    myset = set(res)
    pids = list(myset)
    #Print QC
    print('-'*20)
    print('found the following subjects in folder ...')
    print(pids)
    print('-'*20)
    images_preprocessed=np.empty((0,rows_standard,cols_standard,2))
    masks_preprocessed=np.empty((0,rows_standard,cols_standard,1))

    for f in pids:
    
        FLAIR=f+FLAIR_sub
        FLAIR_image=nb.load(FLAIR)
        FLAIR_array=FLAIR_image.get_data()
        FLAIR_array=np.swapaxes(FLAIR_array,0,2)
        FLAIR_array=np.flip(FLAIR_array,1)
        
        mask=f+mask_sub
        mask_image=nb.load(mask)
        mask_array=mask_image.get_data()
        mask_array=np.swapaxes(mask_array,0,2)
        mask_array=np.flip(mask_array,1)
        
        T1=f+T1_sub
        T1_image=nb.load(T1)
        T1_array=T1_image.get_data()
        T1_array=np.swapaxes(T1_array,0,2)
        T1_array=np.flip(T1_array,1)
        
        label=f+labels_sub
        label_image=nb.load(label)
        label_array=label_image.get_data()
        label_array=np.swapaxes(label_array,0,2)
        label_array=np.flip(label_array,1)
        
        #preprocess images
        imgs_two_channels,label_array,mask_array=preprocessing(FLAIR_array, T1_array, label_array, mask_array)
        
        #save on slice for each subject to QC data (it should be axial!)
        savefig(imgs_two_channels,label_array,mask_array,f)
        
        images_preprocessed=np.concatenate((images_preprocessed, imgs_two_channels), axis=0)
        masks_preprocessed=np.concatenate((masks_preprocessed,label_array ), axis=0)
        print('concatenating ...')
        print(images_preprocessed.shape)
        print(masks_preprocessed.shape)
    
 
    np.save('images_preprocessed.npy',images_preprocessed)
    np.save('masks_preprocessed.npy',masks_preprocessed)

    return

main()