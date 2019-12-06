#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is a repackaging of 'test_your_data.py' script from
https://github.com/hongweilibran/wmh_ibbmTum
Uses a pre-trained MICCAI 2017 U-net winner model to segment WHM on FLAIR(T1) images
As a function running with Keras 2 API. 

Added functionality: 
    include possibility to use a custom made brain mask
    arg parser
    using nibabel to load/save nii (and preserve affine in hdr when saving output)

Tested with TensorFlow 2.0. and keras 2.3.1, python 3.6.8

@author: Marcello Venzi
"""
import copy
import os
import numpy as np
import nibabel as nb
import argparse
import tensorflow as tf
import scipy.spatial
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Cropping2D, ZeroPadding2D
from keras.optimizers import Adam
from keras import backend as K


#%%

def arg_parser():
    parser = argparse.ArgumentParser(description='Use pre-trained MICCAI 2017 U-net to segment WHM on FLAIR(T1) images')
    required = parser.add_argument_group('Required')
    required.add_argument('-f', '--FLAIR', type=str, required=True,
                        help='FLAIR image')

    options = parser.add_argument_group('Options')
    options.add_argument('-t', '--T1', type=str, default=None,
                        help='T1 image (e.g MPRAGE) co-registered to FLAIR')
    options.add_argument('-m', '--mask', type=str, default=None,
                        help='Brain mask (if not provided will mask based on threshold')
    options.add_argument('--model_w_FLAIR', type=str, default='/home/c1025972/Documents/MICCAI/MICCAI_2017_ts2/test_your_data/pretrained_FLAIR_only',
                        help='folder with model weights for single modality CNN')
    options.add_argument('--model_w_FLAIR_T1', type=str, default='/home/c1025972/Documents/MICCAI/MICCAI_2017_ts2/test_your_data/pretrained_FLAIR_T1',
                        help='folder with model weights for dual modality CNN')
    options.add_argument('-o','--outname', type=str, default=None,
                        help='outputname')
    options.add_argument('-p','--printfig', type=bool, default=False,
                        help='outputname')
    return parser


#%% Default parameters

rows_standard = 200  #the input size 
cols_standard = 176
thresh = 30   # threshold for getting the brain mask
per = 0 #Discard slices

#%%

# -preprocessing --------------------------------

def preprocessing(FLAIR_array, T1_array, mask_array=None):
    
    if mask_array is None:
        print('Generating brain mask based on threshold ...')
        mask_array = np.ndarray(np.shape(FLAIR_array), dtype=np.float32)
        mask_array[FLAIR_array >=thresh] = 1
        mask_array[FLAIR_array < thresh] = 0
        for iii in range(np.shape(FLAIR_array)[0]):
            mask_array[iii,:,:] = scipy.ndimage.morphology.binary_fill_holes(mask_array[iii,:,:])  #fill the holes inside brain
    
    FLAIR_array -=np.mean(FLAIR_array[mask_array == 1])      #Gaussian normalization
    FLAIR_array /=np.std(FLAIR_array[mask_array == 1])
    
    rows_o = np.shape(FLAIR_array)[1]
    cols_o = np.shape(FLAIR_array)[2]
    FLAIR_array = FLAIR_array[:, int((rows_o-rows_standard)/2):int((rows_o-rows_standard)/2)+rows_standard, int((cols_o-cols_standard)/2):int((cols_o-cols_standard)/2)+cols_standard]
    
    if T1_array.any():
        T1_array -=np.mean(T1_array[mask_array == 1])      #Gaussian normalisation
        T1_array /=np.std(T1_array[mask_array == 1])
        T1_array = T1_array[:, int((rows_o-rows_standard)/2):int((rows_o-rows_standard)/2)+rows_standard, int((cols_o-cols_standard)/2):int((cols_o-cols_standard)/2)+cols_standard]
    
        imgs_two_channels = np.concatenate((FLAIR_array[..., np.newaxis], T1_array[..., np.newaxis]), axis = 3)
        return imgs_two_channels
    else: 
        return FLAIR_array[..., np.newaxis]

def postprocessing(FLAIR_array, pred):
    start_slice = int(np.shape(FLAIR_array)[0]*per)
    num_o = np.shape(FLAIR_array)[1]  # original size
    rows_o = np.shape(FLAIR_array)[1]
    cols_o = np.shape(FLAIR_array)[2]
    original_pred = np.zeros(np.shape(FLAIR_array), dtype=np.float32)
    original_pred[:,int((rows_o-rows_standard)/2):int((rows_o-rows_standard)/2)+rows_standard,int((cols_o-cols_standard)/2):int((cols_o-cols_standard)/2)+cols_standard] = pred[:,:,:,0]
    original_pred[0: start_slice, ...] = 0
    original_pred[(num_o-start_slice):num_o, ...] = 0
    return original_pred

#%%
# -define u-net architecture--------------------
smooth = 1.
def dice_coef_for_training(y_true, y_pred):
    print(np.shape(y_pred))
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    print(np.shape(y_pred))
    print(np.shape(y_true))
    return -dice_coef_for_training(y_true, y_pred)

def get_crop_shape(target, refer):
        # width, the 3rd dimension
        cw = (target.get_shape()[2] - refer.get_shape()[2])
        assert (cw >= 0)
        if cw % 2 != 0:
            cw1, cw2 = int(cw/2), int(cw/2) + 1
        else:
            cw1, cw2 = int(cw/2), int(cw/2)
        # height, the 2nd dimension
        ch = (target.get_shape()[1] - refer.get_shape()[1])
        assert (ch >= 0)
        if ch % 2 != 0:
            ch1, ch2 = int(ch/2), int(ch/2) + 1
        else:
            ch1, ch2 = int(ch/2), int(ch/2)

        return (ch1, ch2), (cw1, cw2)
          
def get_unet(img_shape = None):
        
        inputs = Input(shape = img_shape)
        concat_axis = -1
            
        conv1 = Conv2D(64, (5, 5), activation='relu', padding='same', data_format="channels_last", name='conv1_1')(inputs)
        conv1 = Conv2D(64, (5, 5), activation='relu', padding='same', data_format="channels_last")(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(conv1)
        conv2 = Conv2D(96, (3, 3), activation='relu', padding='same', data_format="channels_last")(pool1)
        conv2 = Conv2D(96, (3, 3), activation='relu', padding='same', data_format="channels_last")(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(conv2)

        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format="channels_last")(pool2)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format="channels_last")(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(conv3)

        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', data_format="channels_last")(pool3)
        conv4 = Conv2D(256, (4, 4), activation='relu', padding='same', data_format="channels_last")(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(conv4)

        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', data_format="channels_last")(pool4)
        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', data_format="channels_last")(conv5)

        up_conv5 = UpSampling2D(size=(2, 2), data_format="channels_last")(conv5)
        ch, cw = get_crop_shape(conv4, up_conv5)
        crop_conv4 = Cropping2D(cropping=(ch,cw), data_format="channels_last")(conv4)
        up6 = concatenate([up_conv5, crop_conv4], axis=concat_axis)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same', data_format="channels_last")(up6)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same', data_format="channels_last")(conv6)

        up_conv6 = UpSampling2D(size=(2, 2), data_format="channels_last")(conv6)
        ch, cw = get_crop_shape(conv3, up_conv6)
        crop_conv3 = Cropping2D(cropping=(ch,cw), data_format="channels_last")(conv3)
        up7 = concatenate([up_conv6, crop_conv3], axis=concat_axis)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format="channels_last")(up7)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format="channels_last")(conv7)

        up_conv7 = UpSampling2D(size=(2, 2), data_format="channels_last")(conv7)
        ch, cw = get_crop_shape(conv2, up_conv7)
        crop_conv2 = Cropping2D(cropping=(ch,cw), data_format="channels_last")(conv2)
        up8 = concatenate([up_conv7, crop_conv2], axis=concat_axis)
        conv8 = Conv2D(96, (3, 3), activation='relu', padding='same', data_format="channels_last")(up8)
        conv8 = Conv2D(96, (3, 3), activation='relu', padding='same', data_format="channels_last")(conv8)

        up_conv8 = UpSampling2D(size=(2, 2), data_format="channels_last")(conv8)
        ch, cw = get_crop_shape(conv1, up_conv8)
        crop_conv1 = Cropping2D(cropping=(ch,cw), data_format="channels_last")(conv1)
        up9 = concatenate([up_conv8, crop_conv1], axis=concat_axis)
        conv9 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format="channels_last")(up9)
        conv9 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format="channels_last")(conv9)

        ch, cw = get_crop_shape(inputs, conv9)
        conv9 = ZeroPadding2D(padding=(ch, cw), data_format="channels_last")(conv9)
        conv10 = Conv2D(1, (1, 1), activation='sigmoid', data_format="channels_last")(conv9)
        model = Model(inputs=inputs, outputs=conv10)
        model.compile(optimizer=Adam(lr=(1e-4)*2), loss=dice_coef_loss, metrics=[dice_coef_for_training])

        return model

#%%-----------------------------------------------------------------------------------


def savefig(FLAIR_array,original_pred,outf):
    
    #Select slices with highest intensity (more WMH)
    
    zslice=int(np.argmax(np.sum(original_pred[:,:,:],axis=1).sum(axis=1))) #sum over xy for each z -> zslices


    my_cmap = copy.copy(plt.cm.get_cmap('jet')) 
    my_cmap.set_bad(alpha=0) # set NaN to zero alpha
    #Save axial slice with highest detection
    f = plt.figure()
    f.add_subplot(1,2, 2)
    plt.imshow(FLAIR_array[zslice,:,:], cmap='gray')
    
    pred_mask = np.array(original_pred, dtype=float)
    pred_mask[pred_mask < 1] = np.nan

    plt.imshow(pred_mask[zslice,:,:], cmap=my_cmap, alpha=1,vmin=0, vmax=1)
    plt.axis('off')
    f.add_subplot(1,2, 1)
    plt.imshow(FLAIR_array[zslice,:,:], cmap='gray')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    if not os.path.exists(outf):
        outf=''
    plt.savefig(outf+"whm_example_slice_"+str(zslice) +".png",bbox_inches='tight',pad_inches=0.0)    
    
    
    return



#%%
        
    
def main(args=None):

    args = arg_parser().parse_args(args)
    
    FLAIR = args.FLAIR
    
    if not os.path.isfile(FLAIR):
        raise Exception("Input FLAIR file not found")
    
    if args.T1 is None:
        dimI=1
        model_folder=args.model_w_FLAIR
        if not os.path.exists(args.model_w_FLAIR):
            raise Exception("Model weights folder " + str(args.model_w_FLAIR) + " not found")
    else:
        dimI=2
        model_folder=args.model_w_FLAIR_T1
        if not os.path.exists(args.model_w_FLAIR_T1):
            raise Exception("Model weights folder " + str(args.model_w_FLAIR_T1) + " not found")
    
    if args.mask is not None:
        if not os.path.isfile(args.mask):
            raise Exception("Input mask file not found")
        else:
            mask_image=nb.load(args.mask)
            mask_array=mask_image.get_data()
            mask_array=np.swapaxes(mask_array,0,2)
            mask_array=np.flip(mask_array,1)
    else:
        mask_array=None
    
    if args.outname is None:
        args.outname='whm_mask.nii.gz'
        outf=os.getcwd()
    else:
        outf=os.path.abspath(args.mask)
        if os.path.isfile(args.outname):
            print('Warning: overwriting output file : ' + str(args.outname))
    
  
    #------Load data
    FLAIR_image=nb.load(FLAIR)
    FLAIR_array=FLAIR_image.get_data()
    FLAIR_array=np.swapaxes(FLAIR_array,0,2)
    FLAIR_array=np.flip(FLAIR_array,1)
    
    if args.T1 is not None:
        T1_image=nb.load(args.T1)
        T1_array=T1_image.get_data()
        T1_array=np.swapaxes(T1_array,0,2)
        T1_array=np.flip(T1_array,1)
    else:
        T1_array = []
    
    #------Preprocess    
    imgs_test = preprocessing(np.float32(FLAIR_array), np.float32(T1_array),mask_array) 
    
    #------Predict
    img_shape=(rows_standard, cols_standard, dimI)
    model = get_unet(img_shape) 
    model.load_weights(os.path.join(model_folder,'0.h5'))  # 3 ensemble models
    print('-'*30)
    print('Segmenting WHM on 3 ensemble models ...') 
    pred_1 = model.predict(imgs_test, batch_size=1, verbose=1)
    model.load_weights(os.path.join(model_folder, '1.h5')) 
    pred_2 = model.predict(imgs_test, batch_size=1, verbose=1)
    model.load_weights(os.path.join(model_folder, '2.h5'))
    pred_3 = model.predict(imgs_test, batch_size=1, verbose=1)
    pred = (pred_1+pred_2+pred_3)/3   #average prediction of the 3 CNNs
    pred[pred[...,0] > 0.45] = 1      #0.45 thresholding 
    pred[pred[...,0] <= 0.45] = 0
    
    #------Postprocess
    original_pred = postprocessing(FLAIR_array, pred) # get the original size to match
        
    #-----Save image
    if args.printfig:
        print('Saving Figure')
        savefig(FLAIR_array,original_pred,outf)
        
    #------Save output
    original_pred = np.swapaxes(original_pred,0,2)
    hdr = FLAIR_image.header.copy()
    hdr.set_data_dtype(np.int8)
    nb.Nifti1Image(original_pred.astype(np.int8), FLAIR_image.affine, hdr).to_filename(args.outname)
    
    
    return


if __name__=='__main__':
    main()
