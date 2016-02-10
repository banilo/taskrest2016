"""
ARCHI/HCP: decomposing many task and rest maps

Author: Danilo Bzdok, 2014/2015
        danilobzdok[AT]gmail[DOT]com
"""

print __doc__

#autoindent
from os import mkdir, getcwd, path as op
import numpy as np
from nilearn._utils import check_niimg, check_niimgs
from nilearn.plotting import plot_stat_map

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
from nilearn._utils import check_niimgs

# sklearn imports
from sklearn.svm import LinearSVC, SVC
from sklearn import feature_selection
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import (KFold, StratifiedKFold, cross_val_score,
                                      StratifiedShuffleSplit)
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.cluster import WardAgglomeration
from sklearn.feature_extraction import image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import WardAgglomeration

# CONST
memname = 'mygarbage'
mask_image = 'grey10_icbm.nii'
ATLAS_SRC_DIR = op.join(getcwd(), 'atlas_rawdata')
ATLAS_TAR_DIR = op.join(getcwd(), 'atlas_masks')
n_jobs = 4
use_compression = False
SVM_PEN = 'l1'
SMOOTH = 0
N_COMP_CLUST_MULT = 10  # multiplier for clustering compressions
data_path = r'/Volumes/TRESOR/archi/glm/unsmoothed/'
write_dir = ('/git/cohort/archi/compr2task' +
             '_s%i' % SMOOTH + '_svm' + SVM_PEN)


# memoization
from joblib import Memory
memory = Memory(cachedir='cache_is_here', verbose=2)

# @memory.cache
def get_FS_from_paths(path_list, masker):
    print "Loading first-level images from subjects data..."
    l = []
    niis = check_niimgs(path_list)
    data4d = niis.get_data()
    for iimg in range(data4d.shape[3]):
        masked = masker.transform(
            nib.Nifti1Image(data4d[..., iimg], niis.get_affine()))[0]
        l.append(masked)
    FS_org = np.asarray(l)
    return FS_org
    
# write_dir = '/tmp/archi_axial'
if op.exists(write_dir) is False:
    mkdir(write_dir)

subjects = ['sujet_%03d' % i for i in range(1, 9) + range(10, 80)]
n_subjects = len(subjects)

contrasts = [
    'expression_control',
    'expression_intention', 'expression_sex',
    'face_control', 'face_sex', 'face_trusty', 'false_belief_audio',
    'false_belief_video',
    'grasp-orientation', 'hand-side', 'intention-random', 'mecanistic_audio',
    'mecanistic_video', 'motor-cognitive', 'non_speech', 'object_grasp',
    'object_orientation', 'rotation_hand', 'rotation_side', 'saccade',
    'speech', 'triangle_random', 'triangle_intention'
]
n_contrasts = len(contrasts)

# left, right, audio, video, phraseaudio, phrasevideo,     'computation', 'sentences'
stat_paths = np.array([op.join(data_path, subject, '%s_z_map.nii' % beta)
                      for subject in subjects for beta in contrasts])
y_subs = np.array([isub for isub, subject in enumerate(subjects) for beta in contrasts])
y_contr = np.array([icon for subject in subjects for icon, con in enumerate(contrasts)])

firstdataimg = nib.load(stat_paths[0])
masknii = resample_img(mask_image, target_shape=firstdataimg.shape,
                       target_affine=firstdataimg.get_affine(), interpolation='nearest')
mask = masknii.get_data().astype(np.bool)
affine = masknii.get_affine()
header = masknii.get_header()
nifti_masker = NiftiMasker(mask_img=masknii, smoothing_fwhm=False,
                           standardize=False)
nifti_masker.fit()
nifti_masker.mask_img_.to_filename("debug_mask.nii")

from sklearn.decomposition import FactorAnalysis, RandomizedPCA, TruncatedSVD
from sklearn.decomposition import SparsePCA, PCA, FastICA
from nilearn.decomposition.multi_pca import MultiPCA

switch_halves = False

outcome_mat = np.zeros((4, 20))
sparsity_mat = np.zeros((4, 20))
steps = np.arange(100)[::5] + 5
for i_compo, n_components in enumerate(steps):
    compressors = [
        WardAgglomeration(),
        FastICA(n_components=n_components),
        PCA(n_components=n_components, whiten=True),
        SparsePCA(n_components=n_components, alpha=1.0, n_jobs=n_jobs,
                  verbose=1, tol=0.1),
        # FactorAnalysis(n_components=n_components),
    ]

    for i_compr, compressor in enumerate(compressors):
        # first CV: according to SUBJECTS NOT TASKS
        folder = iter(StratifiedShuffleSplit(y_subs, test_size=0.5))
        first_half, second_half = folder.next()
        if switch_halves:
            first_half, second_half = folder.next()
        print "Task distr 1st 1/2: " + str(np.bincount(y_contr[first_half]))
        print "Task distr 2nd 1/2: " + str(np.bincount(y_contr[second_half]))

        # a) non-supervised: find lower dimensions
        FS1 = get_FS_from_paths(stat_paths[first_half], nifti_masker)
        compressor.fit(FS1)

        # b) supervised: test this reduction
        FS2 = get_FS_from_paths(stat_paths[second_half], nifti_masker)
        FS2_reduced = compressor.transform(FS2)
        labels2 = y_contr[second_half]

        n_iter = 100  # Bertrand: 100 is enough
        # second CV: according to TASKS NOT SUBJECTS
        folder2 = StratifiedShuffleSplit(labels2, test_size=0.1, n_iter=n_iter)
        clf1 = LinearSVC(multi_class='ovr', penalty='l2', verbose=0)
        acc_list = []
        coef_list = []
        for (train_inds, test_inds) in folder2:
            clf1.fit(FS2_reduced[train_inds, :], labels2[train_inds])
            pred_y = clf1.predict(FS2_reduced[test_inds, :])
            acc = (pred_y == labels2[test_inds]).mean()
            acc_list.append(acc)
            coef_list.append(clf1.coef_)
            
        # save mean accuracy
        compr_mean_acc = np.mean(acc_list)
        outcome_mat[i_compr, i_compo] = compr_mean_acc
        
        # L1 norm divided by L2 norm of coef matrix (latter is Frobenius)        
        compr_coef = np.asarray(coef_list).mean(axis=0)
        sparsity = np.sum(np.abs(compr_coef)) / linalg.norm(compr_coef,
                                                            ord='fro')
        sparsity_mat[i_compr, i_compo] = sparsity

print "DONE! HAVE FUN."

from joblib import dump
plt.figure()
for i, compressor in enumerate(compressors):
    plt.plot(steps, outcome_mat[i, :],
             label=compressor.__class__.__name__)
plt.title('23-class distinction performance by compressor', {'fontsize': 20})
plt.xlabel('n_components')
plt.ylabel('Multi-class prediction accuracy (LinearSVM)')
plt.yticks(np.linspace(0, 0.9, 17))
plt.xticks(steps)
plt.legend(loc='lower right')
plt.savefig(op.join(write_dir, 'compr_ncomp_acc.png'), dpi=300)

plt.figure()
for i, compressor in enumerate(compressors):
    plt.plot(steps, sparsity_mat[i, :],
             label=compressor.__class__.__name__)
plt.title('Sparsity of compressor coefficients', {'fontsize': 20})
plt.xlabel('n_components')
plt.ylabel('Sparsity (L1/Frobenius)')
plt.yticks(np.linspace(0, 0.9, 17))
plt.xticks(steps)
plt.legend(loc='lower right')
plt.savefig(op.join(write_dir, 'compr_ncomp_sparsity.png'), dpi=300)


