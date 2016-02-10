"""
ARCHI/HCP: decomposing many task and rest maps

Author: Danilo Bzdok, 2014/2015
        danilobzdok[AT]gmail[DOT]com
"""

print __doc__

import os
from os import mkdir, getcwd, path as op
import numpy as np
from nilearn._utils import check_niimg
from nilearn.plotting import plot_stat_map
import glob
import joblib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.close('all')
plt.style.use('ggplot')

import nibabel as nib
from numpy import linalg

# nilearn
from nilearn import datasets, input_data
from nilearn.input_data import NiftiMasker, NiftiLabelsMasker
from xml.etree import ElementTree
from nilearn.image import resample_img, smooth_img

# sklearn imports
from sklearn.svm import LinearSVC, SVC
from sklearn import feature_selection
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import (LeaveOneLabelOut)
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.metrics import accuracy_score, precision_score
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction import image
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neighbors import KNeighborsClassifier

# CONST
mask_image = 'mask_actual_one.nii'


masknii = nib.load(mask_image)
mask = masknii.get_data().astype(np.bool)
target_affine = masknii.get_affine()
target_header = masknii.get_header()
target_shape = masknii.shape
nifti_masker = NiftiMasker(mask_img=masknii, smoothing_fwhm=False,
                           standardize=False)
nifti_masker.fit()
nifti_masker.mask_img_.to_filename("debug_mask.nii")

for compr_name in ['PCA', 'FactorAnalysis',
                   'FastICA', 'SparsePCA']:
    for sample in ['AT', 'HT']:

        FS_nii, labels, subs = joblib.load('preload_2nd_' + sample)

        print('%s in %s' % (compr_name, sample))

        FS = nifti_masker.transform(FS_nii)

        compressor = joblib.load('preload_compr_HT%s40' % compr_name)

        FS_loadings = compressor.transform(FS)
        from scipy.stats import f_oneway

        for group, group_name in zip([labels, subs],
                                     ['tasks', 'subjects']):
            print(group_name)
            array_form = [FS_loadings[group_tag == group].ravel() for group_tag in np.unique(group)]

            fvalue, pvalue = f_oneway(*array_form)
            print('F-Value: %.4f' % fvalue)
            print('P-Value: %.16f' % pvalue)

            
