"""
ARCHI/HCP: decomposing many task and rest maps

Author: Danilo Bzdok, 2014/2015
        danilobzdok[AT]gmail[DOT]com
"""

print __doc__

import os
from os import mkdir, path as op
import numpy as np
from numpy import linalg
from scipy.stats import zscore, scoreatpercentile
import glob

import matplotlib
# matplotlib.style.use('ggplot')
import matplotlib.pyplot as plt
plt.close('all')

import nibabel as nib

# sklearn
from sklearn.svm import LinearSVC
from sklearn.cross_validation import StratifiedShuffleSplit, LeaveOneLabelOut
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.cluster import (SpectralClustering, KMeans, WardAgglomeration)
from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import (
    SparsePCA, PCA, FastICA, IncrementalPCA)
from sklearn.feature_extraction import image
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV

# nilearn
from nilearn.image import resample_img, smooth_img
from nilearn.plotting import plot_stat_map, plot_epi
from nilearn.input_data import NiftiMasker, NiftiLabelsMasker
from nilearn._utils import concat_niimgs, check_niimgs
from nilearn.decomposition import CanICA

from parietal.learn.decompositions.gspca import GSPCAModel

# PySurfer
from surfer import Brain, io

# CONST
memname = 'mygarbage'
# dataset = 'ARCHI'
mask_image = 'grey10_icbm.nii'
MAX_IMAGES_PER_CONTRAST = 5000  # because HTC has more data than ARCHI
n_jobs = 1
plot_figures = True
plot_pysurfer_figs = False  # only relevant if plot_figures == True
PNG_DPI = 200
my_cm10 = [
    '#4BBCF6', '#98E466', '#FBEF69', '#FF0030', '#000000',
    '#A7794F', '#CCCCCC', '#85359C', '#FF9300', '#F47D7D'
]

from joblib import Memory, dump, load
memory = Memory(cachedir='cache_is_here', verbose=2)


# 1. variant: load task FS FILE-WISE
def get_FS_from_paths(path_list, masker):
    print "Loading first-level TASK images from subjects..."
    # n_features = len(np.where(masker.mask_img_.get_data() != 0)[0])
    n_img = len(path_list)
    # FS_org = np.zeros((n_img, n_features))
    # load preciously (half-loaded) FS
    # if ('HCP' in path_list[0]):
    #     TMP_FNAME = 'dump_HCP_task_concat'
    # else:
    #     TMP_FNAME = 'dump_ARCHI_task_concat'
    # if op.exists(TMP_FNAME):
    #     FS_org, iimg = load(TMP_FNAME)
    # else:
    FS_org = [] 
    iimg = 0
    while iimg < len(path_list):  # load images by index
        if (iimg % 25 == 0):
            print "{0}/{1}".format(iimg, n_img)
            # dump((FS_org, iimg), TMP_FNAME)
        r_nii = resample_img(
            path_list[iimg],
            target_shape=masker.mask_img_.shape,
            target_affine=masker.mask_img_.get_affine(),
            interpolation='nearest')
        # masked = masker.transform(r_nii)
        # if FS_org is None:
        #     FS_org = r_nii
        # else:
        #     FS_org = np.vstack((FS_org, r_nii))
        FS_org.append(r_nii)
        iimg += 1
    return concat_niimgs(FS_org)

def get_FS_from_noise(masker):
    # CONST
    n_random_niis = 1000
    n_random_foci = 300, 400  # random amount of foci per study
    fwhm = 12

    print "Inventing first-level NOISE images from Gaussians..."
    # generate some noise niftis
    inds = np.where(masker.mask_img_.get_data())  # grid of locations
    x_inds, y_inds, z_inds = inds  # unpack tuple
    list_niis = []
    for inii in xrange(n_random_niis):
        if inii % 24 == 0:
            print "%i/%i" % (inii + 1, n_random_niis)
        nfoci = np.random.randint(n_random_foci[0], n_random_foci[1])
        cur_img = np.zeros(masker.mask_img_.shape)
        for ifocus in xrange(nfoci):
            # find random point within mask
            i = np.random.randint(0, len(x_inds) - 1)
            x, y, z = x_inds[i], y_inds[i], z_inds[i]
            # put a dot there
            if np.random.randint(0, 2):
                cur_img[x, y, z] = 150
            else:
                cur_img[x, y, z] = -150

        # smooth current image of random foci
        cur_fwhm = np.random.randint(fwhm, fwhm + 6)
        new_nii = smooth_img(
            nib.Nifti1Image(cur_img, masker.mask_img_.get_affine(),
                header=masker.mask_img_.get_header()), cur_fwhm)

        # zscore that
        # img_masked = cur_img[masker.mask_img_.get_data().astype(np.bool)]
        # img_masked = zscore(img_masked)
        # cur_img[masker.mask_img_.get_data().astype(np.bool)] = img_masked

        list_niis.append(new_nii)
    check_niimgs(list_niis).to_filename('debug_DN.nii')
    return concat_niimgs(list_niis)


def get_FS_from_rest(masker, anal_tag):
    COMP_PER_IMG = 20
    RS_TMP_FNAME = 'dump_rs_tmp'
    print "Loading first-level REST images from subjects..."
    if anal_tag == 'HR':  # HCP
        rs_files = glob.glob('/Volumes/TRESOR/neurospin/Volumes/DANILO2/neurospin/population/HCP/S500-1/*/MNINonLinear/Results/rfMRI_REST1_??/rfMRI_REST1_??.nii.gz')
        rs_files = rs_files[:50]
        first_img = nib.load(rs_files[0])
        n_img = len(rs_files) * COMP_PER_IMG
        rs_masker = NiftiMasker(mask_img='grey10_icbm.nii', smoothing_fwhm=8.)
        rs_masker.fit()
        if op.exists(RS_TMP_FNAME):
            reduced_4d_data = load(RS_TMP_FNAME)
        else:
            reduced_4d_data = np.zeros((rs_masker.mask_img_.shape + (n_img, )))
        iimg = 0
        for i_f, f in enumerate(rs_files):
            istart = i_f * COMP_PER_IMG
            if np.any(reduced_4d_data[..., istart] != 0):
                continue
            if (i_f % 5 == 0):
                print "{0}/{1}".format(i_f, len(rs_files))
                dump(reduced_4d_data, RS_TMP_FNAME)
            img = nib.load(f)
            masked = rs_masker.transform(img)
            masked = StandardScaler().fit_transform(masked)
            # rpca = RandomizedPCA(n_components=COMP_PER_IMG, whiten=True)
            rpca = PCA(n_components=COMP_PER_IMG, whiten=True)
            rpca.fit(masked)
            comp_img = rs_masker.inverse_transform(zscore(rpca.components_))
            comp_img.to_filename('debug_rs_pca_comp.nii.gz')
            reduced_4d_data[..., istart:istart + COMP_PER_IMG] = comp_img.get_data()
        dump(reduced_4d_data, RS_TMP_FNAME)
        rs_4d = nib.Nifti1Image(reduced_4d_data,
            affine=first_img.get_affine(),
            header=first_img.get_header())
        rs_4d.to_filename('preproc_RS_1000.nii.gz')

        r_nii = resample_img(
            rs_4d,
            target_shape=masker.mask_img_.shape,
            target_affine=masker.mask_img_.get_affine(),
            interpolation='continuous')

        r_nii.to_filename('res.nii.gz')
        return r_nii
    else:
        raise NotImplementedError

print 'Finding statistical images...'
stat_paths = {}
y_subs = {}
y_contr = {}
contrasts = {}
n_subjects = {}

# ARCHI dataset
cur_key = 'AT'
datapath = '/Volumes/TRESOR/archi/glm2/smoothed'
contrasts[cur_key] = [
    'face_sex', 'face_trusty',
    'false_belief_audio', 'false_belief_video',

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
subjects = ['sujet_%03d' % i for i in range(1, 9) + range(10, 80)]
stat_paths[cur_key] = np.array([op.join(datapath, subject, '%s_z_map.nii' % beta)
                   for subject in subjects for beta in contrasts[cur_key]])
y_subs[cur_key] = np.array([isub for isub, subject in enumerate(subjects) for beta in contrasts[cur_key]])
y_contr[cur_key] = np.array([icon for subject in subjects for icon, con in enumerate(contrasts[cur_key])])

# HACK: account for missing files
inds_found = np.asarray([op.exists(f) for f in stat_paths[cur_key]])
n_misses = len(inds_found) - np.sum(inds_found)
print "{0} contrast images missing!".format(n_misses)

# subjects = np.asarray(subjects)[inds_found]
stat_paths[cur_key] = np.asarray(stat_paths[cur_key])[inds_found]
y_subs[cur_key] = y_subs[cur_key][inds_found]
y_contr[cur_key] = y_contr[cur_key][inds_found]
n_subjects[cur_key] = len(subjects)

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
y_subs[cur_key] = np.asarray(y_subs[cur_key])
y_contr[cur_key] = np.asarray(y_contr[cur_key])
n_subjects[cur_key] = len(np.unique(y_subs[cur_key]))
stat_paths[cur_key] = np.array(stat_paths[cur_key])

n_contrasts = len(contrasts)

target_space_img = '/Volumes/TRESOR/archi/glm2/smoothed/sujet_001/expression_intention-control_z_map.nii'
firstdataimg = nib.load(target_space_img)
masknii = resample_img(mask_image, target_shape=firstdataimg.shape,
                       target_affine=firstdataimg.get_affine(), interpolation='nearest')
mask = masknii.get_data().astype(np.bool)
target_affine = masknii.get_affine()
target_header = masknii.get_header()
target_shape = masknii.shape
nifti_masker = NiftiMasker(mask_img=masknii, smoothing_fwhm=False,
                           standardize=False)
nifti_masker.fit()
nifti_masker.mask_img_.to_filename("debug_mask.nii")

# H=HCP, A=ARCHI, T=task, R=rest, DN=data noise
analysis_modes = [
    'AT-AT', 'HT-AT', 'HR-AT', 'DN-AT',
    'HT-HT', 'AT-HT', 'HR-HT', 'DN-HT'
]

# n_comp_list = [2, 3, 4] + list(np.arange(5, 105, 5))

n_comp_list = [40] # [20, 10, 5, 1]
SVM_PEN_list = ['l1']

mat_shape = (len(SVM_PEN_list), len(analysis_modes), len(n_comp_list), 8)
if op.exists('mat_actual_nparc'):
    mat_actual_nparc = load('mat_actual_nparc')
    mat_acc = load('mat_acc')
    mat_prec = load('mat_prec')
    mat_rec = load('mat_rec')
    mat_spars = load('mat_spars')
else:
    mat_actual_nparc = np.ones(mat_shape) * -1
    mat_acc = np.ones(mat_shape) * -1
    mat_prec = np.ones(mat_shape) * -1
    mat_rec = np.ones(mat_shape) * -1
    mat_spars = np.ones(mat_shape) * -1
for i_SVM_PEN, SVM_PEN in enumerate(SVM_PEN_list):
    for i_anal_mode, analysis_mode in enumerate(analysis_modes):
        anal_1st, anal_2nd = analysis_mode.split('-')
        write_dir = ('/git/cohort/archi/compr2task' + '_%s' % analysis_mode +
                     '_svm' + SVM_PEN)
        if op.exists(write_dir) is False:
            mkdir(write_dir)

        for i_comp, n_components in enumerate(n_comp_list):
            compressors = [

                # compression by source variation
                FastICA(n_components=n_components, whiten=True),
                
                SparsePCA(n_components=n_components, alpha=1.0,  # big sparsity
                          n_jobs=n_jobs, verbose=10, tol=0.1),
                       
                           
                FactorAnalysis(n_components=n_components),
                
                IncrementalPCA(n_components=n_components, whiten=True,
                               batch_size=100),
                
                # compression by topographic (=flat) clustering
                WardAgglomeration(n_clusters=n_components),

                KMeans(n_clusters=n_components, max_iter=100,
                       n_jobs=1)
            ]

            for i_compr, compressor in enumerate(compressors):

                # if mat_actual_nparc[i_SVM_PEN, i_anal_mode, i_comp, i_compr] != -1:
                #     continue  # this one already been calculated

                plt.close('all')
                if hasattr(compressor, 'n_clusters'):
                    actual_nparcel = compressor.n_clusters
                else:
                    actual_nparcel = int(n_components)  # needed for the clusterings
                fname_suffix = ''
                compr_name = compressor.__class__.__name__

                print '-' * 80
                print '%s/%s: n=%i' % (analysis_mode, compr_name, actual_nparcel)
                print '-' * 80

                # first CV: partition according to SUBJECTS, not tasks
                first_half, second_half = {}, {}
                for k in y_subs.keys():
                    n_subs = len(np.unique(y_subs[k]))
                    # folder = iter(StratifiedShuffleSplit(y_subs[k], test_size=0.5,
                    #               random_state=0))  # needed for disk-loaded FS
                    sub_ids1 = np.unique(y_subs[k])[(n_subs / 2):]
                    sub_ids2 = np.unique(y_subs[k])[:(n_subs / 2)]
                    a = np.in1d(y_subs[k], sub_ids1)
                    b = np.in1d(y_subs[k], sub_ids2)
                    a_inds = np.where(a == True)[0]
                    b_inds = np.where(b == True)[0]
                    first_half[k] = a_inds
                    second_half[k] = b_inds
                    # sanity checks for subject partition in data splits
                    assert(n_subs == len(sub_ids1) + len(sub_ids2))
                    assert(len(a) == len(np.unique(list(a_inds) +
                           list(b_inds))))  # every index allocated?
                    assert(np.all(np.bincount(list(first_half[k]) +
                           list(second_half[k])) == 1))

                # a) non-supervised: find lower dimensional representation
                if (anal_1st[1] != 'R') and (anal_1st[1] != 'N'):
                    try:
                        print "Task distr 1st 1/2: " + str(np.bincount(y_contr[anal_1st][first_half[anal_1st]]))
                        print "Task distr 2nd 1/2: " + str(np.bincount(y_contr[anal_1st][second_half[anal_1st]]))
                    except:
                        pass

                anal_1st_dump = op.join(os.getcwd(), 'preload_1st_' + anal_1st)
                if op.exists(anal_1st_dump):
                    FS1 = load(anal_1st_dump)
                    print "DATA LOADED FROM DISK!"
                else:
                    if anal_1st[1] == 'T':
                        FS1 = get_FS_from_paths(
                            stat_paths[anal_1st][first_half[anal_1st]],
                            nifti_masker)
                    elif anal_1st[:2] == 'DN':
                        FS1 = get_FS_from_noise(nifti_masker)
                    elif anal_1st[1] == 'R':
                        FS1 = get_FS_from_rest(nifti_masker, anal_1st)
                    else:
                        raise NotImplementedError
                    dump(FS1, anal_1st_dump)  # persistence
                
                if (anal_1st != 'DN') and (anal_1st[-1] != 'R'):
                    subs1 = y_subs[anal_1st][first_half[anal_1st]]
                    labels1 = y_contr[anal_1st][first_half[anal_1st]]

                print "n_samples={0}, n_features={1}".format(
                    FS1.shape[-1], np.prod(FS1.shape[0:3]))

                print "Learning compression..."
                dump_compr_fname = ('preload_compr_' + anal_1st +
                    compr_name + (n_components).__str__())
                if op.exists(dump_compr_fname):
                    print 'COMPRESSION LOADED FROM DISK!'
                    compressor = load(dump_compr_fname)
                    if 'KMeans' in compr_name:
                        km_label_masker = load(dump_compr_fname + '_labels')
                else:
                    if compr_name == 'WardAgglomeration':
                        # cluster
                        connectivity = image.grid_to_graph(n_x=target_shape[0],
                                                           n_y=target_shape[1],
                                                           n_z=target_shape[2], mask=mask)
                        compressor = WardAgglomeration(n_clusters=actual_nparcel,
                                                       connectivity=connectivity,
                                                       memory=memory)
                        # sample_inds = np.random.randint(0, FS1.shape[0], 100)
                        compressor.fit(nifti_masker.transform(FS1))
                    elif 'KMeans' in compr_name:  # might be MiniBatch variant
                        # cluster
                        # sample_inds = np.random.randint(0, FS1.shape[0], 50)
                        compressor.fit(
                            nifti_masker.transform(FS1).T)  # cluster the mask voxels, not the images!
                        km_labels_img = nifti_masker.inverse_transform(
                            compressor.labels_ + 1)  # transform 0 label by incrementation
                        km_label_masker = NiftiLabelsMasker(
                            labels_img=km_labels_img, background_label=0,
                            standardize=False, detrend=False,
                            mask_img=nifti_masker.mask_img_)
                        km_label_masker.fit()
                    elif compr_name == 'CanICA':
                        # niis_per_sub = [nib.Nifti1Image(FS1.get_data()[..., subs1 == s],
                        #     nifti_masker.mask_img_.get_affine()) for s in np.unique(subs1)]
                        # 
                        # # make sure each subjects split contains ALL 18 tasks
                        # n_imgs_per_sub = [niis_per_sub[i].shape[-1] for i in range(len(niis_per_sub))]
                        # n_imgs_per_sub = np.array(n_imgs_per_sub)
                        # inds_good = n_imgs_per_sub >= n_components
                        # inds_bad = n_imgs_per_sub < n_components
                        # print "CanICA: skipping %i task/subject-splits!!" % (np.sum(inds_bad))
                        # niis_per_sub = np.array(niis_per_sub)[inds_good]
                        # 
                        # compressor.fit(niis_per_sub)
                        compressor.fit(FS1)
                    elif compr_name == 'GSPCAModel':
                        # data_per_sub = [nifti_masker.transform(
                        #     nib.Nifti1Image(FS1.get_data()[..., subs1 == s],
                        #     nifti_masker.mask_img_.get_affine())) for s in np.unique(subs1)]
                        # compressor.fit(data_per_sub)
                        compressor.fit([nifti_masker.transform(FS1)])
                        compressor.components_ = compressor.maps_
                    else:
                        FS1_trans = nifti_masker.transform(FS1)
                        compressor.fit(FS1_trans)

                    # persistance
                    dump(compressor, dump_compr_fname)
                    if 'KMeans' in compr_name:
                        dump(km_label_masker, dump_compr_fname + '_labels')

                if compr_name == 'FactorAnalysis':
                    FS1_trans = nifti_masker.transform(FS1)
                    mean_ = FS1_trans.mean(axis=0)  # needed for FA

                # import pdb; pdb.set_trace()

                del FS1  # free memory

                # b) supervised: test this reduction
                print "Compressing new data using %i components/clusters..." %\
                    actual_nparcel
                anal_2nd_dump = op.join(os.getcwd(), 'preload_2nd_' + anal_2nd)
                if op.exists(anal_2nd_dump):
                    (FS2, labels2, subs2) = load(anal_2nd_dump)
                    print "DATA LOADED FROM DISK!"
                else:
                    FS2 = get_FS_from_paths(
                        stat_paths[anal_2nd][second_half[anal_2nd]],
                        nifti_masker)

                    labels2 = y_contr[anal_2nd][second_half[anal_2nd]]
                    subs2 = y_subs[anal_2nd][second_half[anal_2nd]]

                    dump((FS2, labels2, subs2), anal_2nd_dump)  # persistence
                labels2 = np.asarray(labels2)
                subs2 = np.asarray(subs2)

                if 'KMeans' in compr_name:  # might be MiniBatch variant
                    FS2_reduced = km_label_masker.transform(FS2)
                elif compr_name == 'GSPCAModel':
                    # l = [n[0] for n in FS2]
                    FS2_masked = nifti_masker.transform(FS2)
                    # compute the projections on the components by hand
                    from sklearn.linear_model import ridge_regression
                    ridge_alpha = 0.01

                    U = ridge_regression(compressor.components_.T, FS2_masked.T,
                                         ridge_alpha, solver='cholesky')
                    s = np.sqrt((U ** 2).sum(axis=0))
                    s[s == 0] = 1
                    U /= s
                    FS2_reduced = U
                elif compr_name == 'CanICA':
                    FS2_reduced = compressor.transform([FS2])
                    FS2_reduced = FS2_reduced[0]
                else: # (Incremental)PCA, Ward, FA
                    FS2_reduced = compressor.transform(nifti_masker.transform(FS2))

                ss = StandardScaler()
                FS2_reduced = ss.fit_transform(FS2_reduced)

                # compute "proportion of explained variance"
                # FS2_total_var = np.sum(np.var(FS2, axis=0))
                # if compr_name == 'PCA':  # we have whitened !!
                #     FS2_expl_var = np.sum(compressor.explained_variance_)
                # elif compr_name == 'FastICA':
                #     ridge_alpha = 0.01
                # else:
                #     FS2_expl_var = np.var(FS2_reduced, axis=0)
                # prop_expl_var = FS2_expl_var / FS2_total_var
                # print "Proportion of explained variance: " + prop_expl_var

                # second CV: according to TASK-SET PER SUBJECTS
                # rationale: avoid that (different) task images from same
                # subject appear in train and in test set

                folder2 = LeaveOneLabelOut(subs2)

                print "Learning important loadings..."
                params = {}
                params['C'] = np.logspace(-3, 3, 7)
                clf1 = LinearSVC(multi_class='ovr', penalty=SVM_PEN, verbose=0,
                                 dual=False)
                coef_list = []
                acc_list = []
                cm_list = []
                prfs_list = []
                for (train_inds, test_inds) in folder2:
                    print "Grid-searching optimal C hyper-parameter..."
                    bestclf = GridSearchCV(clf1, param_grid=params,
                                           n_jobs=n_jobs)
                    bestclf.fit(FS2_reduced[train_inds, :], labels2[train_inds])
                    for str in bestclf.grid_scores_:
                        print str
                    print "Setting C to %.2f at %d%%" % (
                        bestclf.best_params_['C'], bestclf.best_score_ * 100)
                    pred_y = bestclf.predict(FS2_reduced[test_inds, :])

                    acc = (pred_y == labels2[test_inds]).mean()
                    cm = confusion_matrix(pred_y, labels2[test_inds])
                    if (cm.shape != (18, 18)):
                        # HACK: a fold occurred that does not contain all 18
                        # tasks
                        continue

                    acc_list.append(acc)
                    coef_list.append(bestclf.best_estimator_.coef_)
                    cm_list.append(cm)
                    prfs_list.append(precision_recall_fscore_support(
                                     labels2[test_inds], pred_y))
                                     
                compr_mean_acc = np.mean(acc_list)
                compr_coef = np.asarray(coef_list).mean(axis=0)
                compr_cm = np.asarray(cm_list).mean(axis=0)
                prfs = np.asarray(prfs_list).mean(axis=0)
                # L1 norm divided by L2 norm of coef matrix (latter is Frobenius)
                sparsity = np.sum(np.abs(compr_coef)) / linalg.norm(compr_coef,
                                                                    ord='fro')

                # regularization paths
                print "Learning regularization paths..."
                cur_title = '%s (n_comp=%s) with %.2f total acc' % (
                    compressor.__class__.__name__,
                    actual_nparcel,
                    compr_mean_acc)
                for PEN in ['l1', 'l2']:
                    folder2p = LeaveOneLabelOut(subs2)
                    C_grid = np.logspace(-7, 7, 15)
                    coef_list2 = []
                    acc_list2 = []
                    train_inds, test_inds = iter(folder2p).next()
                    for my_C in C_grid:
                        print "C: %.4f" % my_C
                        clf2 = LinearSVC(multi_class='ovr', penalty=PEN,
                                         verbose=0,
                                         dual=False, C=my_C)
                        clf2.fit(FS2_reduced[train_inds, :], labels2[train_inds])
                        pred_y = clf2.predict(FS2_reduced[test_inds, :])

                        acc = (pred_y == labels2[test_inds]).mean()

                        coef_list2.append(clf2.coef_)
                        acc_list2.append(acc)

                    coef_list2 = np.array(coef_list2)
                    acc_list2 = np.array(acc_list2)

                    # plot paths
                    n_cols = 5
                    n_rows = 4

                    f, axarr = plt.subplots(nrows=n_rows, ncols=n_cols,
                        figsize=(15, n_rows * 2.2), facecolor='white')
                    t, i_row, i_col = 0, 0, 0
                    for t in np.arange(18):
                        i_row, i_col = divmod(t, n_cols)
                        for n in np.arange(n_components):
                            axarr[i_row, i_col].plot(np.log(C_grid),
                                coef_list2[:, t, n], label='component %i' % (n + 1),
                                color=my_cm10[n % 10], linewidth=3)
                        if i_col == 0:
                            axarr[i_row, i_col].set_ylabel('Coefficients')
                        else:
                            handle = axarr[i_row, i_col].get_yticklabels()
                            plt.setp(handle, visible=False)
                        if (t > 14):
                            axarr[i_row, i_col].set_xlabel('ln(C)')
                        else:
                            handle = axarr[i_row, i_col].get_xticklabels()
                            plt.setp(handle, visible=False)
                        axarr[i_row, i_col].set_title(contrasts[anal_2nd][t])
                        if t == 17:  # take legend of last plot
                            axarr[i_row, i_col].legend(
                                bbox_transform=plt.gcf().transFigure, fontsize=12,
                                bbox_to_anchor=(.86, .1, .1, .2))

                    axarr[3, 4].axis('off')
                    axarr[3, 3].plot(np.log(C_grid), acc_list2, color='#000000',
                                     linewidth=1.5)
                    axarr[3, 3].set_title('ACCURACY', color='r')
                    axarr[3, 3].set_ylim(0, 1.05)
                    axarr[3, 3].set_yticks(np.linspace(0, 1, 11))
                    axarr[3, 3].set_xlabel('ln(C)')
                    plt.suptitle(PEN + '-Regularization path: ' + cur_title,
                                 fontsize=18)
                    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.9,
                                        hspace=0.2, wspace=0.2)
                    plt.savefig(
                        op.join(write_dir, cur_title + '_path_' + PEN + '.png'),
                            dpi=PNG_DPI, facecolor='white')

                # select k best: ACROSS TASKS
                from sklearn.pipeline import make_pipeline
                from sklearn.feature_selection import SelectKBest, f_classif
                from sklearn.multiclass import OneVsRestClassifier

                myKs = [1, 5, 10, 20, 40]
                best_k_container = {}
                best_k_container['acc'] = np.ones((len(myKs))) * -1
                best_k_container['se'] = np.ones((len(myKs))) * -1
                for iK, K in enumerate(myKs):
                    print 'Univariate feature selection: K=%i' % K
                    # selection + classification pipeline
                    clf2 = LinearSVC(multi_class='ovr', penalty='l2',
                                     verbose=0, dual=False, C=1.0)
                    wrap_clf = OneVsRestClassifier(clf2)
                    n_folds = 100
                    folder3 = StratifiedShuffleSplit(
                        labels2, n_folds, test_size=0.1, random_state=0)
                    pipe = make_pipeline(
                        SelectKBest(f_classif, K), wrap_clf)

                    acc_list_k = []
                    sel_frequ = np.zeros((n_components))
                    for train, test in folder3:
                        # k
                        pipe.fit(
                            FS2_reduced[train, :],
                            labels2[train])
                        y_pred = pipe.predict(FS2_reduced[test, :])
                        acc_list_k.append(np.mean(y_pred == labels2[test]))
                        
                        sk = pipe.steps[0][1]
                        sel_frequ += sk.get_support()
                        
                    sel_frequ /= n_folds
                    
                    best_k_container['acc'][iK] = np.mean(acc_list_k)
                    best_k_container['se'][iK] = (
                        np.std(acc_list_k) / np.sqrt(len(acc_list_k)))
                    best_k_container[K] = sel_frequ

                # plot classification scores at selected K
                plt.figure()
                plt.errorbar(myKs, best_k_container['acc'],
                             best_k_container['se'])
                plt.xticks(myKs)
                plt.legend(loc='lower right', fontsize=8)
                plt.ylim(0, 1.)
                plt.yticks(np.linspace(0, 1., 11))
                plt.xlabel('#best components (SelectKBest)')
                plt.ylabel('mean accuracy (+/- SE)')
                
                plt.title(('Best K ACROSS tasks (SVM-l2 C=1.0) in %s: ' %
                             anal_2nd) + cur_title, fontsize=11)
                plt.savefig(op.join(write_dir, cur_title + '_kbest_acrosstask.png'),
                            dpi=PNG_DPI)
                dump_fname = op.join(write_dir, 'dump_kbest_acrosstask_' + compr_name +
                                     n_components.__str__())
                dump(best_k_container, dump_fname)
                
                # plot chosen components
                f, axarr = plt.subplots(nrows=len(myKs[:4]), ncols=1,
                                        figsize=(8, 10))
                for i_K, K in enumerate(myKs[:4]):
                    im = axarr[i_K].imshow(
                        np.tile(best_k_container[K], 18).reshape(18, 40),
                        interpolation='nearest',
                        cmap=plt.cm.Reds,
                        vmax=1.)
                    axarr[i_K].tick_params(
                        labelbottom='on',labeltop='off', labelright='off')
                    axarr[i_K].set_xticks(np.arange(4, n_components + 1, 5))
                    axarr[i_K].set_xticklabels(
                        np.arange(4, n_components + 1, 5) + 1)
                    axarr[i_K].set_yticks(np.arange(0, 18, 2))
                    axarr[i_K].set_yticklabels(
                        np.arange(0, 18, 2) + 1)
                    axarr[i_K].set_ylabel('tasks')
                    axarr[i_K].set_title('%i best' % K)

                # plt.colorbar(im)
                plt.xlabel('selected components')
                plt.subplots_adjust(hspace=0.4)
                plt.suptitle(('Best K ACROSS tasks (SVM-l2 C=1.0) in %s: ' %
                             anal_2nd) + cur_title, fontsize=11)
                plt.savefig(op.join(write_dir, cur_title + '_kbest_acrosstask_grid.png'),
                            dpi=PNG_DPI)

                # select k best: PER TASKS
                from sklearn.pipeline import make_pipeline
                from sklearn.feature_selection import SelectKBest, f_classif
                from sklearn.multiclass import OneVsRestClassifier

                myKs = [1, 5, 10, 20, 40]
                best_k_container = {}
                best_k_container['acc'] = np.ones((len(myKs))) * -1
                best_k_container['se'] = np.ones((len(myKs))) * -1
                best_k_container['sel_frequ'] = np.zeros(
                    (len(myKs), 18, n_components))
                for iK, K in enumerate(myKs):
                    print 'Univariate feature selection: K=%i' % K

                    # selection + classification pipeline
                    clf2 = LinearSVC(multi_class='ovr', penalty='l2',
                                     verbose=0, dual=False, C=1.0)
                    n_folds = 100
                    folder3 = StratifiedShuffleSplit(
                        labels2, n_folds, test_size=0.1, random_state=0)
                    pipe = make_pipeline(
                        SelectKBest(f_classif, K), clf2)
                    pipe = OneVsRestClassifier(pipe)

                    acc_list_k = []
                    sel_frequ = np.zeros((n_components))
                    for train, test in folder3:
                        # k
                        pipe.fit(
                            FS2_reduced[train, :],
                            labels2[train])
                        y_pred = pipe.predict(FS2_reduced[test, :])
                        acc_list_k.append(np.mean(y_pred == labels2[test]))
                        
                        sk = pipe.estimator.steps[0][1]
                        
                        for i_est in np.arange(18):
                            s = pipe.estimators_[i_est].steps[0][1].get_support()
                            best_k_container['sel_frequ'][iK, i_est] += s
                    
                    best_k_container['acc'][iK] = np.mean(acc_list_k)
                    best_k_container['se'][iK] = (
                        np.std(acc_list_k) / np.sqrt(len(acc_list_k)))

                best_k_container['sel_frequ'] /= n_folds  # normalize all

                # plot classification scores at selected K
                plt.figure()
                plt.errorbar(myKs, best_k_container['acc'],
                             best_k_container['se'])
                plt.xticks(myKs)
                plt.legend(loc='lower right', fontsize=8)
                plt.ylim(0, 1.)
                plt.yticks(np.linspace(0, 1., 11))
                plt.xlabel('#best componenents (SelectKBest)')
                plt.ylabel('mean accuracy (+/- SE)')
                
                plt.title(('Best K PER tasks (SVM-l2 C=1.0) in %s: ' %
                             anal_2nd) + cur_title, fontsize=11)
                plt.savefig(op.join(write_dir, cur_title + '_kbest_pertask.png'),
                            dpi=PNG_DPI)

                dump_fname = op.join(write_dir, 'dump_kbest_pertask_' + compr_name +
                                     n_components.__str__())
                dump(best_k_container, dump_fname)

                # plot chosen components
                f, axarr = plt.subplots(nrows=len(myKs[:4]), ncols=1,
                                        figsize=(8, 10))
                for i_K, K in enumerate(myKs[:4]):
                    im = axarr[i_K].imshow(
                        best_k_container['sel_frequ'][i_K, :, :],
                        interpolation='nearest',
                        cmap=plt.cm.Reds,
                        vmax=1.)
                    axarr[i_K].tick_params(
                        labelbottom='on',labeltop='off', labelright='off')
                    axarr[i_K].set_xticks(np.arange(4, n_components + 1, 5))
                    axarr[i_K].set_xticklabels(
                        np.arange(4, n_components + 1, 5) + 1)
                    axarr[i_K].set_yticks(np.arange(0, 18, 2))
                    axarr[i_K].set_yticklabels(
                        np.arange(0, 18, 2) + 1)
                    axarr[i_K].set_ylabel('tasks')
                    axarr[i_K].set_title('%i best' % K)

                # plt.colorbar(im)
                plt.xlabel('selected components')
                plt.subplots_adjust(hspace=0.4)
                plt.suptitle(('Best K PER tasks (SVM-l2 C=1.0) in %s: ' %
                             anal_2nd) + cur_title, fontsize=11)
                plt.savefig(op.join(write_dir, cur_title + '_kbest_pertask_grid.png'),
                            dpi=PNG_DPI)

                # plot f-tests
                # plt.figure(figsize=(8, 8))
                # plt.imshow(
                #     -np.log10(p_list),
                #     interpolation='nearest',
                #     cmap=plt.cm.Reds,
                #     vmax=10)
                # plt.xticks(range(actual_nparcel), np.arange(actual_nparcel) + 1)
                # plt.yticks(range(len(contrasts[anal_2nd])), contrasts[anal_2nd])
                # plt.xlabel('-log10(p) for each component/class')
                # plt.colorbar()
                # plt.title(('One-versus-Rest F-tests in %s: ' % anal_2nd) +
                #           cur_title)
                # plt.savefig(op.join(write_dir, cur_title + '_ftest.png'),
                #             dpi=PNG_DPI)

                # store results
                mat_actual_nparc[i_SVM_PEN, i_anal_mode, i_comp, i_compr] = actual_nparcel
                mat_acc[i_SVM_PEN, i_anal_mode, i_comp, i_compr] = compr_mean_acc
                mat_prec[i_SVM_PEN, i_anal_mode, i_comp, i_compr] = np.mean(prfs[0])
                mat_rec[i_SVM_PEN, i_anal_mode, i_comp, i_compr] = np.mean(prfs[1])
                mat_spars[i_SVM_PEN, i_anal_mode, i_comp, i_compr] = sparsity
                dump(mat_actual_nparc, 'mat_actual_nparc')
                dump(mat_acc, 'mat_acc')
                dump(mat_prec, 'mat_prec')
                dump(mat_rec, 'mat_rec')
                dump(mat_spars, 'mat_spars')

                if plot_figures:
                    plt.figure(figsize=(7.5, 7))
                    # compr_coef_abs = np.abs(compr_coef)
                    masked_data = np.ma.masked_where(compr_coef != 0., compr_coef)
                    plt.imshow(masked_data, interpolation='nearest', cmap=plt.cm.gray_r)
                    masked_data = np.ma.masked_where(compr_coef == 0., compr_coef)
                    plt.imshow(masked_data, interpolation='nearest', cmap=plt.cm.RdBu_r)
                    
                    plt.xticks(range(actual_nparcel), np.arange(actual_nparcel) + 1)
                    plt.yticks(range(len(contrasts[anal_2nd])), contrasts[anal_2nd])

            
                    plt.xlabel('weights per component (sparsity: %.2f)' %
                               sparsity)
                    plt.ylabel('tasks')
                    plt.grid(False)
                    plt.colorbar()
                    plt.title(cur_title, {'fontsize': 16})
                    plt.savefig(op.join(write_dir, cur_title + '.png'), dpi=PNG_DPI)
                    plt.show()

                    # print precision/recall
                    plt.figure(figsize=(13, 11))
                    plt.plot(range(len(contrasts[anal_2nd])),prfs[0], label='precision')
                    plt.plot(range(len(contrasts[anal_2nd])),prfs[1], label='recall')
                    plt.xticks(range(len(contrasts[anal_2nd])), contrasts[anal_2nd], rotation=90)
                    plt.ylabel('Precision')
                    plt.title(cur_title + ': Precision', {'fontsize': 16})
                    plt.ylim(0, 1.)
                    plt.legend(loc='lower right')
                    plt.savefig(op.join(write_dir, cur_title + '_precrec.png'), dpi=PNG_DPI,
                                facecolor='white')

                    # print confusion matrix
                    plt.figure(figsize=(13, 11))
                    plt.imshow(compr_cm, interpolation='nearest',
                               cmap=plt.cm.Reds)
                    plt.title(cur_title + ': Confusion matrix', {'fontsize': 16})
                    plt.colorbar()
                    plt.grid(False)
                    plt.ylabel('True label')
                    plt.yticks(range(len(contrasts[anal_2nd])), contrasts[anal_2nd])
                    plt.xlabel('Predicted label')
                    plt.xticks(range(len(contrasts[anal_2nd])), contrasts[anal_2nd], rotation=90)
                    plt.savefig(op.join(write_dir, cur_title + '_cm.png'), dpi=PNG_DPI,
                                facecolor='white')

                    plt.figure(figsize=(13, 11))
                    sc = SpectralClustering(n_clusters=3)
                    sc.fit(compr_cm)
                    new_order = np.argsort(sc.labels_)
                    compr_cm_r = compr_cm[new_order, :][:, new_order]
                    contr_r = np.asarray(contrasts[anal_2nd])[new_order]
                    plt.imshow(compr_cm_r, interpolation='nearest',
                               cmap=plt.cm.Reds)
                    plt.title(cur_title + ': Confusion matrix (ordered)', {'fontsize': 16})
                    plt.colorbar()
                    plt.grid(False)
                    plt.ylabel('True label')
                    plt.yticks(range(len(contrasts[anal_2nd])), contr_r)
                    plt.xlabel('Predicted label')
                    plt.xticks(range(len(contrasts[anal_2nd])), contr_r, rotation=90)
                    plt.savefig(op.join(write_dir, cur_title + '_cm_r.png'), dpi=PNG_DPI,
                                facecolor='white')

                    # print correlation coefficits
                    if (n_components != 1):
                        cc = np.corrcoef(compr_coef)

                        plt.figure(figsize=(13, 11))
                        plt.imshow(cc, interpolation='nearest',
                                   cmap=plt.cm.RdBu_r, vmax=1.0, vmin=-1.0)
                        plt.title(cur_title + ': Correlation matrix', {'fontsize': 16})
                        plt.colorbar()
                        plt.grid(False)
                        plt.ylabel('True label')
                        plt.yticks(range(len(contrasts[anal_2nd])), contrasts[anal_2nd])
                        plt.xlabel('Predicted label')
                        plt.xticks(range(len(contrasts[anal_2nd])), contrasts[anal_2nd], rotation=90)
                        plt.savefig(op.join(write_dir, cur_title + '_cc.png'), dpi=PNG_DPI,
                                    facecolor='white')

                        sc = SpectralClustering(n_clusters=3)
                        sc.fit(cc)
                        new_order = np.argsort(sc.labels_)
                        cc_r = cc[new_order, :][:, new_order]
                        contr_r = np.asarray(contrasts[anal_2nd])[new_order]
                        plt.figure(figsize=(13, 11))
                        plt.imshow(cc_r, interpolation='nearest',
                                   cmap=plt.cm.RdBu_r, vmax=1.0, vmin=-1.0)
                        plt.title(cur_title + ': Correlation matrix (ordered)',
                                  {'fontsize': 16})
                        plt.colorbar()
                        plt.grid(False)
                        plt.ylabel('True label')
                        plt.yticks(range(len(contrasts[anal_2nd])), contr_r)
                        plt.xlabel('Predicted label')
                        plt.xticks(range(len(contrasts[anal_2nd])), contr_r, rotation=90)
                        plt.savefig(op.join(write_dir, cur_title + '_cc_r.png'), dpi=PNG_DPI,
                                    facecolor='white')

                    # plot components in 3D
                    if hasattr(compressor, 'components_'):
                        my_min = +0.5
                        my_max = +2
                        for i in range(actual_nparcel):
                            print "Rendering component %i..." % (i + 1)
                            fname_prefix = (compressor.__class__.__name__ + '_c%i-%i' % (
                                            actual_nparcel, i + 1))
                            fname = fname_prefix + fname_suffix
                            comp_masked = compressor.components_[i]
                            if not 'KMeans' in compr_name:
                                comp_masked = zscore(comp_masked)
                            comp_img = nifti_masker.inverse_transform(comp_masked)
                            comp_fname = op.join(write_dir, fname + '_comp.nii.gz')
                            comp_img.to_filename(comp_fname)

                            if plot_pysurfer_figs:
                                brain = Brain("fsaverage", "split", "inflated",
                                              views=['lat', 'med'],
                                              config_opts=dict(background="black"))

                                mri_file = comp_fname
                                reg_file = "/Applications/freesurfer/average/mni152.register.dat"
                                surf_data_lh = io.project_volume_data(mri_file, "lh", reg_file,
                                                                      target_subject='fsaverage')
                                surf_data_rh = io.project_volume_data(mri_file, "rh", reg_file,
                                                                      target_subject="fsaverage")

                                brain.add_overlay(surf_data_lh, hemi='lh', sign='pos', min=my_min,
                                                  max=my_max, name='blah1')
                                brain.add_overlay(surf_data_lh, hemi='lh', sign='neg', min=my_min,
                                                  max=my_max, name='blah2')

                                brain.add_overlay(surf_data_rh, hemi='rh', sign='pos', min=my_min,
                                                  max=my_max, name='blah3')
                                brain.add_overlay(surf_data_rh, hemi='rh', sign='neg', min=my_min,
                                                  max=my_max, name='blah4')

                                brain.save_image(mri_file.split('.nii')[0] + fname_suffix + '.png')

                    # plot components in 2D
                    if hasattr(compressor, 'components_'):
                        # components
                        n_cols = 5
                        n_rows = (actual_nparcel / n_cols) + (1 if
                                                            (actual_nparcel % n_cols != 0) else 0)
                        plt.figure(figsize=(17, n_rows * 2.2), facecolor='black')
                        plt.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0)
                        for i in range(actual_nparcel):
                            fig_id = plt.subplot(n_rows, n_cols, i + 1)
                            comp_3D = nifti_masker.inverse_transform(
                                compressor.components_[i, :])
                            plot_stat_map(comp_3D, 'colin.nii', annotate=False,
                                          draw_cross=False,
                                          title='Comp%i' % (i + 1), axes=fig_id,
                                          colorbar=False)
                            plt.savefig(op.join(write_dir, cur_title + '_comp.png'), dpi=PNG_DPI,
                                        facecolor='black')
                    else:
                        # clusters
                        if compr_name == 'WardAgglomeration':
                            clust_img = nifti_masker.inverse_transform(
                                compressor.labels_.astype(np.float))
                        elif 'KMeans' in compr_name:  #  might be MiniBatch variant
                            clust_img = km_label_masker.labels_img_

                        clust_img.to_filename(op.join(write_dir, cur_title + '_clust.nii.gz'))
                        plot_epi(clust_img, cut_coords=(45, 28, 5),
                                 title=cur_title, display_mode='ortho', draw_cross=False)
                        plt.savefig(op.join(write_dir, cur_title + '_clust.png'),
                                    dpi=PNG_DPI, facecolor='black')

                    # archetypical reconstruction images
                    for i, contr_name in enumerate(contrasts[anal_2nd]):
                        print "Rendering task archetyp %i: %s..." % ((i + 1), contr_name)
                        fname_prefix = (compressor.__class__.__name__ +
                                        '_c%i_%s_archetyp' % (actual_nparcel, contr_name))
                        fname = fname_prefix + fname_suffix

                        if hasattr(compressor, 'components_'):
                            # components
                            if ((compr_name == 'SparsePCA') or
                               (compr_name == 'GSPCAModel')):
                                decomp_masked = np.dot(compr_coef[i, :],
                                                       compressor.components_)
                            elif compr_name == 'FactorAnalysis':
                                decomp_masked = np.dot(compr_coef[i, :],
                                                       compressor.components_) + mean_
                            elif compr_name == 'CanICA':  # HACK???
                                # decomp = compressor.inverse_transform(
                                #     compr_coef[i, :])
                                # decomp_masked = nifti_masker.transform(
                                #     decomp[0])[0, :]
                                
                                decomp_masked = np.dot(
                                    compr_coef[i, :], compressor.components_)
                                
                            else:  # FastICA
                                decomp_masked = compressor.inverse_transform(
                                    compr_coef[i, :])
                        else:
                            # clusters
                            if compr_name == 'WardAgglomeration':
                                decomp_masked = compressor.inverse_transform(
                                    compr_coef[i, :]).astype(np.float)
                            elif 'KMeans' in compr_name:  # might be MiniBatch variant
                                learned_loadings = compr_coef[i, :].reshape(1, actual_nparcel)
                                decomp_img = km_label_masker.inverse_transform(
                                    learned_loadings)
                                decomp_masked = nifti_masker.transform(decomp_img)
                            else:
                                raise NotImplementedError

                        # my_min = scoreatpercentile(np.abs(decomp_masked), 5)
                        # my_max = scoreatpercentile(np.abs(decomp_masked), 95)
                        my_min = 0.1
                        my_max = 3

                        # reshaping row vectors into column vectors
                        if decomp_masked.shape[0] == 1:
                            decomp_masked = decomp_masked[0]

                        # dump to disk
                        decomp_masked = zscore(decomp_masked)
                        decomp_img = nifti_masker.inverse_transform(decomp_masked)
                        decomp_fname = op.join(write_dir, fname + '.nii.gz')
                        decomp_img.to_filename(decomp_fname)

                        if plot_pysurfer_figs:
                            try:
                                brain = Brain("fsaverage", "split", "inflated",
                                              views=['lat', 'med'],
                                              config_opts=dict(background="black"))
                            
                                mri_file = decomp_fname
                                reg_file = "/Applications/freesurfer/average/mni152.register.dat"
                                surf_data_lh = io.project_volume_data(mri_file, "lh", reg_file,
                                                                      target_subject='fsaverage')
                                surf_data_rh = io.project_volume_data(mri_file, "rh", reg_file,
                                                                      target_subject="fsaverage")
                            
                                brain.add_overlay(surf_data_lh, hemi='lh', sign='pos', min=my_min,
                                                  max=my_max, name='blah1')
                                brain.add_overlay(surf_data_lh, hemi='lh', sign='neg', min=my_min,
                                                  max=my_max, name='blah2')
                            
                                brain.add_overlay(surf_data_rh, hemi='rh', sign='pos', min=my_min,
                                                  max=my_max, name='blah3')
                                brain.add_overlay(surf_data_rh, hemi='rh', sign='neg', min=my_min,
                                                  max=my_max, name='blah4')
                            
                                brain.save_image(mri_file.split('.nii')[0] + fname_suffix + '.png')
                            except:
                                pass
print "DONE."