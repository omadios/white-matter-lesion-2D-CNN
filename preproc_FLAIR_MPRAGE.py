#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Preprocess a pair of MPRAGE and  FLAIR images: skullstrip, register (Rigid), N4 bias field correction.
Output is bias-corrected FLAIR and bias-corrected MPRAGE (registered to FLAIR)

@author: Marcello Venzi
"""

import copy
import os
import argparse
from shutil import copyfile

from nipype.interfaces.ants.segmentation import BrainExtraction
from nipype.interfaces.ants import Registration
from nipype.interfaces.ants import N4BiasFieldCorrection
from nipype.interfaces.ants import ApplyTransforms
from nipype.interfaces import afni


#%%


def arg_parser():
    parser = argparse.ArgumentParser(description='Register, skullstrip and bias correct pairs of T1 and FLAIR images ')
    required = parser.add_argument_group('Required')
    required.add_argument('-t', '--T1', type=str, required=True,
                        help='T1 image (e.g MPRAGE)')
    required.add_argument('-f', '--FLAIR', type=str, required=True,
                        help='FLAIR image')
    required.add_argument('-p', '--prefix', type=str, required=True,
                        help='prefix for filename (e.g. subject id)')

    options = parser.add_argument_group('Options')
    options.add_argument('-m', '--mask', type=str, default=None,
                        help='T1 mask (if not provided, will run ants brain extraction)')
    options.add_argument('--storetemp', type=str, default=False,
                        help='keep temporary files')
    options.add_argument('--brain_template', type=str, default='T_template0.nii.gz',
                        help='brain template for brain extraction')
    options.add_argument('--temp_prob', type=str, default='T_template0_BrainCerebellumProbabilityMask.nii.gz',
                        help='brain probability mask for brain extraction')
    options.add_argument('--temp_mask', type=str, default='T_template0_BrainCerebellumRegistrationMask.nii.gz',
                        help='template mask for brain extraction')
    options.add_argument('-o','--outfolder', type=str, default=None,
                        help='folder where to store output')
    return parser



#%% add arg parser

def main(args=None):

    args = arg_parser().parse_args(args)
    FLAIR = args.FLAIR
    MPRAGE = args.T1
    
    prefix=args.prefix + '.'

    if args.mask is None:
        args.temp_mask = os.path.abspath (args.temp_mask)
        args.brain_template = os.path.abspath(args.brain_template)
        args.temp_prob = os.path.abspath(args.temp_prob)
        if not os.path.isfile(args.temp_mask):
            raise Exception("template mask not foud")
        if not os.path.isfile(args.brain_template):
            raise Exception("brain template mask not foud")
        if not os.path.isfile(args.temp_prob):
            raise Exception("template probability mask not foud")
    elif not os.path.isfile(args.mask):
            raise Exception("T1 mask file not foud")

    if not os.path.isfile(MPRAGE):
        raise Exception("Input T1 file not found")
    if not os.path.isfile(FLAIR):
        raise Exception("Input FLAIR file not found")

    if args.outfolder is not None:
        abs_out = os.path.abspath(args.outfolder)
        #print(abs_out)
        if not os.path.exists(abs_out):
            #if selecting a new folder copy the files (not sure how to specify different folder under nipype when it runs sh scripts from ants)
            os.mkdir(abs_out)
        copyfile(os.path.abspath(MPRAGE),os.path.join(abs_out,os.path.basename(MPRAGE)))
        copyfile(os.path.abspath(FLAIR),os.path.join(abs_out,os.path.basename(FLAIR)))
        if args.mask is not None:
            if os.path.isfile(args.mask):
                copyfile(os.path.abspath(args.mask),os.path.join(abs_out, prefix + 'MPRAGE.mask.nii.gz'))
        os.chdir(args.outfolder)
    elif args.mask is not None:
        copyfile(os.path.abspath(args.mask),os.path.join(os.path.abspath(args.mask), prefix + 'MPRAGE.mask.nii.gz'))

    if args.mask is None:
        # T1 brain extraction
        brainextraction = BrainExtraction()
        brainextraction.inputs.dimension = 3
        brainextraction.inputs.anatomical_image = MPRAGE
        brainextraction.inputs.brain_template = args.brain_template
        brainextraction.inputs.brain_probability_mask = args.temp_prob
        brainextraction.inputs.extraction_registration_mask= args.temp_mask
        brainextraction.inputs.debug=True
        print("brain extraction")
        print(' ')
        print(brainextraction.cmdline)
        print('-'*30)
        brainextraction.run()
        os.rename('highres001_BrainExtractionMask.nii.gz',prefix +'MPRAGE.mask.nii.gz')
        os.rename('highres001_BrainExtractionBrain.nii.gz',prefix +'MPRAGE.brain.nii.gz')
        os.remove('highres001_BrainExtractionPrior0GenericAffine.mat')
        os.rmdir('highres001_')

    #two step registration with ants (step1)

    reg = Registration()
    reg.inputs.fixed_image = FLAIR
    reg.inputs.moving_image = MPRAGE
    reg.inputs.output_transform_prefix = "output_"
    reg.inputs.output_warped_image = prefix + 'output_warped_image.nii.gz'
    reg.inputs.dimension = 3
    reg.inputs.transforms = ['Rigid']
    reg.inputs.transform_parameters = [[0.1]]
    reg.inputs.radius_or_number_of_bins = [32]
    reg.inputs.metric = ['MI']
    reg.inputs.sampling_percentage = [0.1]
    reg.inputs.sampling_strategy = ['Regular']
    reg.inputs.shrink_factors = [[4,3,2,1]]
    reg.inputs.smoothing_sigmas = [[3,2,1,0]]
    reg.inputs.sigma_units = ['vox']
    reg.inputs.use_histogram_matching = [False]
    reg.inputs.number_of_iterations = [[1000,500,250,100]]
    reg.inputs.winsorize_lower_quantile = 0.025
    reg.inputs.winsorize_upper_quantile = 0.975
    print("first pass registration")
    print(' ')
    print(reg.cmdline)
    print('-'*30)
    reg.run()

    os.rename('output_0GenericAffine.mat',prefix + 'MPRAGE_to_FLAIR.firstpass.mat')

    #apply tranform MPRAGE mask to FLAIR

    at = ApplyTransforms()
    at.inputs.dimension = 3
    at.inputs.input_image = prefix + 'MPRAGE.mask.nii.gz'
    at.inputs.reference_image = FLAIR
    at.inputs.output_image = prefix + 'FLAIR.mask.nii.gz'
    at.inputs.interpolation = 'MultiLabel'
    at.inputs.default_value = 0
    at.inputs.transforms = [ prefix + 'MPRAGE_to_FLAIR.firstpass.mat']
    at.inputs.invert_transform_flags = [False]
    print("apply stranform to T1 maks")
    print(' ')
    print(at.cmdline)
    print('-'*30)    
    at.run()

    # bias correct FLAIR and MPRAGE

    n4m = N4BiasFieldCorrection()
    n4m.inputs.dimension = 3
    n4m.inputs.input_image = MPRAGE
    n4m.inputs.mask_image = prefix + 'MPRAGE.mask.nii.gz'
    n4m.inputs.bspline_fitting_distance = 300
    n4m.inputs.shrink_factor = 3
    n4m.inputs.n_iterations = [50,50,30,20]
    n4m.inputs.output_image = prefix + 'MPRAGE.N4.nii.gz'
    print("bias correcting T1")
    print(' ')
    print(n4m.cmdline)
    print('-'*30)
    n4m.run()

    n4f = copy.deepcopy(n4m)
    n4f.inputs.input_image = FLAIR
    n4f.inputs.mask_image = prefix + 'FLAIR.mask.nii.gz'
    n4f.inputs.output_image = prefix + 'FLAIR.N4.nii.gz'
    print("bias correcting FLAIR")
    print(' ')
    print(n4f.cmdline)
    print('-'*30)
    n4f.run()

    # mask bias corrected FLAIR and MPRAGE

    calc = afni.Calc()
    calc.inputs.in_file_a = prefix + 'FLAIR.N4.nii.gz'
    calc.inputs.in_file_b = prefix + 'FLAIR.mask.nii.gz'
    calc.inputs.expr='a*b'
    calc.inputs.out_file = prefix +  'FLAIR.N4.masked.nii.gz'
    calc.inputs.outputtype = 'NIFTI'
    calc.inputs.overwrite = True
    calc.run()

    calc1= copy.deepcopy(calc)
    calc1.inputs.in_file_a = prefix + 'MPRAGE.N4.nii.gz'
    calc1.inputs.in_file_b = prefix + 'MPRAGE.mask.nii.gz'
    calc1.inputs.out_file = prefix +  'MPRAGE.N4.masked.nii.gz'
    calc1.inputs.overwrite = True
    calc1.run()

    #register bias corrected

    reg1 = copy.deepcopy(reg)
    reg1.inputs.output_transform_prefix = "output_"
    reg1.inputs.output_warped_image = prefix + 'output_warped_image.nii.gz'
    reg1.inputs.initial_moving_transform = prefix +'MPRAGE_to_FLAIR.firstpass.mat'
    print("second pass registration")
    print(' ')
    print(reg1.cmdline)
    print('-'*30)
    reg1.run()
    os.rename('output_0GenericAffine.mat',prefix +'MPRAGE_to_FLAIR.secondpass.mat')
    
    
    #generate final mask in FLAIR space

    atf = ApplyTransforms()
    atf.inputs.dimension = 3
    atf.inputs.input_image = prefix + 'MPRAGE.N4.nii.gz'
    atf.inputs.reference_image = FLAIR
    atf.inputs.output_image = prefix + 'MPRAGE.N4.toFLAIR.nii.gz'
    atf.inputs.interpolation = 'BSpline'
    atf.inputs.interpolation_parameters = (3,)
    atf.inputs.default_value = 0
    atf.inputs.transforms = [prefix +  'MPRAGE_to_FLAIR.secondpass.mat']
    atf.inputs.invert_transform_flags = [False]
    print("final apply transform")
    print(' ')
    print(atf.cmdline)
    print('-'*30)
    atf.run()


    #cleanup

    os.remove(prefix + 'output_warped_image.nii.gz')

    if args.outfolder is not None:
        os.remove(os.path.join(abs_out,os.path.basename(MPRAGE)))
        os.remove(os.path.join(abs_out,os.path.basename(FLAIR)))
        
    if args.mask is None:
        os.remove(prefix + 'MPRAGE.brain.nii.gz')
        
    if not args.storetemp:
        os.remove(prefix + 'MPRAGE.mask.nii.gz')
        os.remove(prefix + 'MPRAGE_to_FLAIR.firstpass.mat')
        os.remove(prefix + 'FLAIR.N4.masked.nii.gz')
        os.remove(prefix + 'MPRAGE.N4.masked.nii.gz')
        os.remove(prefix + 'MPRAGE.N4.nii.gz')


    return



main()