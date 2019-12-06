# wmh_MICCAI_tf2
Re-implementation of the winning method in MICCAI 2017 WMH segmentation challenge in TensorFlow2  (https://github.com/hongweilibran/wmh_ibbmTum)

### Description

This repository contains my re-emplemenation of the github code for segmentation White matter hyperintensities (WMH) on FLAIR/T1 mri data. The purpose is to use a CNN approach to segment WHM on FLAIR/T1 data acquired on a 3T Prisma Scanner with 1mm isotropic resolution. The scripts can be run using command line arguments so that they can be intergrated in MRI data analysis pipelines.

The main changes are the following: 
- Code is now compatible with TensorFlow2 and Keras 2.3.1
- Added a preprocessing script to register a pair FLAIR and T1 images using Nipype (calling Ants for skullstripping/registration/Bias correction) that accepts command line arguments
- Added a script to segment preprocessed data using pre-trained ensemble neural networks (added plotting functionality to QC the data)
- Modified training script to that weights avaialble from the original repository can be updated (in my case I retrained the networks using manually labeled data from 10 participants)
- Added a script to concatenate and QC training examples


### Instructions

All code runs in Python 3. In order to run the code the following libraries and MRI tool are required:
```
Requirements: 
Keras 2.3.1, TensorFlow 2.0, Python 3.6.8, h5py, scipy, nipype (Ants/Afni), nibabel 
```

