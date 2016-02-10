"""
ARCHI/HCP: decomposing many task and rest maps

Author: Danilo Bzdok, 2014/2015
        danilobzdok[AT]gmail[DOT]com
"""

print __doc__

#autoindent
import os
from os import mkdir, getcwd, path as op
import numpy as np
import glob

from nilearn._utils import check_niimg, check_niimgs
from nilearn.plotting import plot_stat_map

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.close('all')
plt.style.use('ggplot')

import nibabel as nib
from numpy import linalg

from scipy.stats import ttest_rel

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
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.cluster import WardAgglomeration
from sklearn.feature_extraction import image
from sklearn.neighbors import KNeighborsClassifier

import joblib

# CONST
summaries_dir = '/git/cohort/archi/summaries'
memname = 'mygarbage'
DPI_PNG = 200
mask_image = 'grey10_icbm.nii'
ATLAS_SRC_DIR = op.join(getcwd(), 'atlas_rawdata')
ATLAS_TAR_DIR = op.join(getcwd(), 'atlas_masks')
n_jobs = 4
use_compression = False

# memoization
from joblib import Memory
memory = Memory(cachedir='cache_is_here', verbose=2)

@memory.cache
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

# Get the fMRIdata
reconstr_dir = '/git/cohort/archi/'
write_dir = '/git/cohort/archi/compr2task_means'
# write_dir = '/tmp/archi_axial'
if op.exists(write_dir) is False:
    mkdir(write_dir)

subjects = ['sujet_%03d' % i for i in range(1, 9) + range(10, 80)]
n_subjects = len(subjects)

data_path = r'/Volumes/TRESOR/archi/glm2/smoothed/'
stat_paths = {}
y_subs = {}
y_contr = {}
contrasts = {}
n_subjects = {}

# ARCHI dataset
cur_key = 'AT'
contrasts[cur_key] = [
    # 'expression_intention',
    # 'expression_sex',
    'face_sex', 'face_trusty',
    'false_belief_audio', 'false_belief_video',
    # 'triangle_random', 'triangle_intention',

    'object_grasp', 'object_orientation',
    'rotation_hand',
    'rotation_side',
    'motor-cognitive',
    'left-right', 'right-left',
    'saccade',

    'speech',
    'reading-visual',
    'sentences-computation',
    'computation-sentences',
    'audio-video', 'video-audio'
]

"""
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
"""
# left, right, audio, video, phraseaudio, phrasevideo,     'computation', 'sentences'
stat_paths[cur_key] = np.array([op.join(data_path, subject, '%s_z_map.nii' % beta)
                   for subject in subjects for beta in contrasts[cur_key]])
# y_subs[cur_key] = np.array([isub for isub, subject in enumerate(subjects) for beta in contrasts[cur_key]])
y_contr[cur_key] = np.array([icon for subject in subjects for icon, con in enumerate(contrasts[cur_key])])

# HCP task dataset
cur_key = 'HT'
datapath = '/Volumes/TRESOR/storage/Volumes/DANILO2/storage/workspace/elvis/HCP500_GLM/*/*/z_maps/z_%s*.nii'
contrasts[cur_key] = [
    'REWARD-PUNISH', 'PUNISH-REWARD', 'SHAPES-FACES', 'FACES-SHAPES',
    'RANDOM-TOM', 'TOM-RANDOM',

    'MATH-STORY', 'STORY-MATH',
    'T-AVG', 'F-H', 'H-F',
    'MATCH-REL', 'REL-MATCH',

    'BODY-AVG', 'FACE-AVG', 'PLACE-AVG', 'TOOL-AVG',
    '2BK-0BK'
]
stat_paths[cur_key] = []
for c in contrasts[cur_key]:
    c_paths = np.asarray(glob.glob(datapath % c))
    print "Constrast {0}: {1} images".format(c, len(c_paths))
    stat_paths[cur_key] += c_paths

y_subs[cur_key] = []
y_contr[cur_key] = []
for p in stat_paths[cur_key]:
    index = p.find('HCP500_GLM')
    splits = p[index:].split(os.sep)
    str_task = splits[-1][2:][:-4]
    y_subs[cur_key].append(int(splits[2]))
    i_contr = contrasts[cur_key].index(str_task)  # ValueError if not present
    y_contr[cur_key].append(i_contr)

y_contr[cur_key] = np.asarray(y_contr[cur_key])
stat_paths[cur_key] = np.array(stat_paths[cur_key])

analysis_modes = [
    'AT-AT', 'HT-AT', 'HR-AT', 'DN-AT',
    'HT-HT', 'AT-HT', 'HR-HT', 'DN-HT'
]
PEN = 'svml1'
compr_names = [
    'SparsePCA', 'PCA', 'FastICA', 'FactorAnalysis', 'Ward', 'KMeans'
]


from nilearn.image import mean_img
from surfer import Brain, io
from scipy.stats import pearsonr
from nilearn.input_data import NiftiMasker

masker = NiftiMasker(mask_img='debug_mask.nii').fit()
target_study = ['AT', 'HT']
print "Voxels in GM mask: %i" % \
    len(np.where(masker.mask_img_.get_data() != 0)[0])

if op.exists('dump_mat_rho'):
    mat_rho = load('dump_mat_rho')
else:
    # init dicts
    mat_rho = {}
    for t in target_study:
        mat_rho[t] = np.ones((len(compr_names), len(contrasts[t]), 4)) * -1
    mat_L1 = {}
    for t in target_study:
        mat_L1[t] = np.ones((len(compr_names), len(contrasts[t]), 4)) * -1
        
    # compute + render mean images
    for i_key, key in enumerate(target_study):
        for i, contr_name in enumerate(contrasts[key]):
            # compute mean
            print contr_name
            mri_file = op.join(write_dir, 'mean_' + contr_name + '.nii.gz')
            cur_imgs = stat_paths[key][y_contr[key] == i]
            mean_nifti = mean_img(cur_imgs)
            mean_nifti.to_filename(mri_file)
            
            print "comparing %s with..." % mri_file
            for i_compr, compr_name in enumerate(compr_names):
                for i_ana in range(4):
                    fname_tag = compr_name + '*' + contr_name + '*.nii*'
                    search_str = op.join(reconstr_dir, 'compr2task_' +
                        analysis_modes[4 * i_key + i_ana] + '_' + PEN, fname_tag)
                    l = glob.glob(search_str)
                    print "- %s" % l
                    
                    arr1 = masker.transform(mean_nifti)[0]
                    arr2 = masker.transform(l[0])[0]
                    
                    rho, p = pearsonr(arr1, arr2)
                    mat_rho[key][i_compr, i, i_ana] = rho
                    
                    L1 = np.sum(np.abs(arr1 - arr2))
                    mat_L1[key][i_compr, i, i_ana] = L1
            print '-' * 80

            # render mean
            # my_min = 0.1
            # my_max = 1.5
            # brain = Brain("fsaverage", "split", "inflated",
            #               views=['lat', 'med'],
            #               config_opts=dict(background="black"))
            # reg_file = "/Applications/freesurfer/average/mni152.register.dat"
            # surf_data_lh = io.project_volume_data(mri_file, "lh", reg_file,
            #                                       target_subject='fsaverage')
            # surf_data_rh = io.project_volume_data(mri_file, "rh", reg_file,
            #                                       target_subject="fsaverage")

            # brain.add_overlay(surf_data_lh, hemi='lh', sign='pos', min=my_min, max=my_max,
            #                   name='blah1')
            # brain.add_overlay(surf_data_lh, hemi='lh', sign='neg', min=my_min, max=my_max,
            #                   name='blah2')

            # brain.add_overlay(surf_data_rh, hemi='rh', sign='pos', min=my_min, max=my_max,
            #                   name='blah3')
            # brain.add_overlay(surf_data_rh, hemi='rh', sign='neg', min=my_min, max=my_max,
            #                   name='blah4')

            # brain.save_image(mri_file.split('.nii')[0] + '.png')
    print "DONE! HAVE FUN."
a = b

# radial 
import joblib

import plotly.plotly as py
from plotly.graph_objs import *
mat_rho = joblib.load('dump_mat_rho')
key = 'HT'
i_key = 1
i_ana = 0


i_compr = 2
compr = compr_names[i_compr]

py.sign_in('Python-Demo-Account', 'gwt101uhh0')

rings = []
# colors = ['rgb(106,81,163)', 'rgb(158,154,200)', 'rgb(203,201,226)',
#           'rgb(242,240,247)']
# colors = ['rgb(14, 70, 104)', 'rgb(19,134,191)', 'rgb(22,174,217)',
#           'rgb(206,230,242)']
colors = ['rgb(1, 4, 64)', 'rgb(7, 136, 217)', 'rgb(7, 176, 242)',
          'rgb(242, 62, 22)']
for i_ana in range(4):
    tick_labels = [str(t + 1) for t in np.arange(18)]
    ring_values = mat_rho[key][i_compr, :, i_ana]

    cur_ring = Area(
        r=ring_values,
        t=tick_labels,
        name=analysis_modes[i_ana + i_key * 4],
        marker=Marker(
            color=colors[i_ana]
        )
    )
    rings.append(cur_ring)


data = Data(rings)
layout = Layout(
    title = compr,
    font = Font(
        size=20
    ),
    legend=Legend(
        font=Font(
            size=16
        ),
        # traceorder = 'reversed'
        traceorder = 'normal'
    ),
    radialaxis=RadialAxis(
        range = [0, 1.],
        showticklabels = False
    ),
    angularaxis=AngularAxis(
        tickcolor='rgb(103, 103, 103)'
    ),
    orientation=270,
    xaxis=XAxis(
        ticks='',
        showticklabels=False
    ),
    yaxis=YAxis(
        ticks='',
        showticklabels=False
    ),
)
fig = Figure(data=data, layout=layout)
plot_url = py.plot(fig, filename='polar-area-chart')

# plot single-correlation summaries on network-network similarities
plt.close('all')
for i_compr, compr_name in enumerate(compr_names):
    fig = plt.figure(figsize=(9, 9))
    for i_key, key in enumerate(target_study):
        for i_ana in range(4):
            fig = plt.subplot(4, 2, 1 + i_ana * 2 + i_key)
            
            plt.bar(range(len(contrasts[key])),
                mat_rho[key][i_compr, :, i_ana])
            # ticks only at bottom to safe space
            # if (i_ana == 2) or (i_ana == 5):
            #     plt.xticks(np.arange(len(contrasts[key])) + 0.4,
            #         contrasts[key], rotation=90)
            plt.xticks(np.arange(len(contrasts[key])) + 0.4,
                np.arange(len(contrasts[key])) + 1)
            plt.ylim(0.,1)
            str_title = '%s' % (analysis_modes[4 * i_key + i_ana])
            plt.title(str_title)
            plt.ylabel('rho')

            if i_ana > 0:
                anastr1 = analysis_modes[4 * i_key + i_ana]
                anastr2 = analysis_modes[4 * i_key + i_ana - 1]
                
                a = mat_rho[key][i_compr, :, i_ana]
                b = mat_rho[key][i_compr, :, i_ana - 1]
                t, p = ttest_rel(a, b)

                plt.text(0.3, 0.8,
                    'Two-sided, dep. t-test between %s and %s: \n'
                    'p=%.4f T=%.4f' % (anastr1, anastr2, p, t),
                    fontdict={'fontsize': 8})

        plt.xlabel('Tasks')
    plt.suptitle(compr_names[i_compr], fontsize=20)
    plt.subplots_adjust(left=0.15, right=0.92, bottom=0.1, top=0.9, hspace=0.4)
    png_name = op.join(summaries_dir, 'corr_%s' % compr_name)
    plt.savefig(png_name, dpi=DPI_PNG)

a = z
# all-in-one bar plot
plt.close('all')
b_width = 0.5
my_colors = ['#580F19', '#32658D', '#BF3346', '#51A2E0', '#008E74', '#B2E097']
for i_key, key in enumerate(target_study):
    fig = plt.figure(figsize=(7, 8))
    # plt.xticks(range(3 * 6))
    my_ticks = []
    for i_ana in range(4):
        my_ticks += list(b_width / 2 + np.arange(0., 3., 0.5) + i_ana * 4)
        for i_compr, compr_name in enumerate(compr_names):
            
            values = mat_rho[key][i_compr, :, i_ana]
            cur_mean = np.mean(values)
            cur_std = np.std(values, dtype=np.float64)
            tick_pos = i_compr * (b_width) + i_ana * 4
            if i_ana == 0:
                plt.bar(tick_pos, cur_mean, yerr=cur_std,
                    width=b_width, color=my_colors[i_compr], label=compr_name)
            else:
                plt.bar(tick_pos, cur_mean, yerr=cur_std,
                    width=b_width, color=my_colors[i_compr])
    # plt.xticks(my_ticks, compr_names * 3, rotation=90)
    plt.xticks(np.asarray([b_width * 3] * 4) + np.asarray([0, 4, 8, 12]),
        np.asarray(analysis_modes)[np.arange(4) + i_key * 4])
    plt.ylim(0.,1)
    plt.yticks(np.linspace(0., 1., 11))
    plt.title(key + ": comparison of image reconstruction versus ground truth\n"
                    "mean correlation across 18 tasks",
        fontsize=15)
    plt.ylabel('mean rho (+/-std)')
    plt.legend(loc='upper right')
    plt.show()

    png_name = op.join(summaries_dir, 'corr_mean_%s' % key)
    plt.savefig(png_name, dpi=DPI_PNG)

# print stats
for i_compr, compr_name in enumerate(compr_names):
    key = 'AT'
    print compr_name + '  ' + key
    for i_ana1, i_ana2 in [[0, 1], [0, 2], [1, 2]]:
        print "Paired t-test between %s and %s: " % (
            analysis_modes[i_ana1], analysis_modes[i_ana2])
        fig = plt.subplot(3, 2, 1 + i_ana * 2 + i_key)
        
        a = mat_rho[key][i_compr, :, i_ana1]
        b = mat_rho[key][i_compr, :, i_ana2]
        t, p = ttest_rel(a, b)
        print "p-value: %.4f T value: %.4f" % (p, t)

    print '-' * 40

    key = 'HT'
    print compr_name + '  ' + key
    for i_ana1, i_ana2 in [[0, 1], [0, 2], [1, 2]]:
        print "Paired t-test between %s and %s: " % (
            analysis_modes[i_ana1], analysis_modes[i_ana2])
        fig = plt.subplot(3, 2, 1 + i_ana * 2 + i_key)
        
        a = mat_rho[key][i_compr, :, i_ana1]
        b = mat_rho[key][i_compr, :, i_ana2]
        t, p = ttest_rel(a, b)
        print "p-value: %.4f T value: %.4f" % (p, t)

    print '-' * 80

for i_compr, compr_name in enumerate(compr_names):
    plt.figure(figsize=(9, 9))
    for i_key, key in enumerate(target_study):
        for i_ana in range(3):
            fig = plt.subplot(3, 2, 1 + i_ana * 2 + i_key)
            
            plt.bar(range(len(contrasts[key])),
                mat_L1[key][i_compr, :, i_ana] * -1)
            # ticks only at bottom to safe space
            # if (i_ana == 2) or (i_ana == 5):
            #     plt.xticks(np.arange(len(contrasts[key])) + 0.4,
            #         contrasts[key], rotation=90)
            plt.xticks(np.arange(len(contrasts[key])) + 0.4,
                np.arange(len(contrasts[key])) + 1)
            # plt.ylim(0.,1)
            str_title = '%s' % (analysis_modes[3*i_key + i_ana])
            plt.title(str_title)
            plt.ylabel('- L1 norm')
        plt.xlabel('Tasks')
    plt.suptitle(compr_names[i_compr], fontsize=20)
    png_name = op.join(summaries_dir, 'l1_%s' % compr_name)
    plt.savefig(png_name, dpi=DPI_PNG)
