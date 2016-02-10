"""
ARCHI/HCP: decomposing many task and rest maps

Author: Danilo Bzdok, 2014/2015
        danilobzdok[AT]gmail[DOT]com
"""

from os import path as op
import os
import numpy as np
from joblib import load

import matplotlib
matplotlib.style.use('ggplot')
import matplotlib.pyplot as plt
plt.close('all')

PNG_DPI = 200

write_dir = '/git/cohort/archi/summaries'
if op.exists(write_dir) is False:
    os.mkdir(write_dir)

target_studies = ['AT', 'HT']
analysis_modes = [
    'AT-AT', 'HT-AT','HR-AT', 'DN-AT',
    'HT-HT', 'AT-HT', 'HR-HT','DN-HT'
]
SVM_PEN_list = ['l1', 'l2']
compressors = [
    'PCA', 'SparsePCA', 'FA', 'ICA', 'Ward', 'KMeans'
]

mat_actual_nparc = load('nmat_actual_nparc')
mat_acc = load('nmat_acc')
mat_prec = load('nmat_prec')
mat_rec = load('nmat_rec')
mat_spars = load('nmat_spars')

n_items = 21
i_SVM_PEN = 0

# accuracies
for i_key, key in enumerate(target_studies):
    plt.figure(figsize=(8, 8))
    for i_ana in np.arange(4):
        fig_id = plt.subplot(4, 1, 1 + i_ana)
        my_ticks = mat_actual_nparc[i_SVM_PEN, i_ana + i_key * 4, :, 0][:n_items]
        plt.xticks(my_ticks.astype(np.int))
        for i_compr, compr in enumerate(compressors):        
            plt.plot(
                my_ticks,
                mat_acc[i_SVM_PEN, i_ana + i_key * 4, :, i_compr][:n_items],
                label=compr)
        plt.ylabel('Accuracy')
        plt.title(analysis_modes[i_ana + i_key * 4] +
                  '/SVM-' + SVM_PEN_list[i_SVM_PEN])
        # plt.ylim(0, 1.)
        plt.yticks(np.linspace(0, 1, 11))
        plt.legend(loc='lower right', fontsize=8)
    plt.xlabel('#Components')
    plt.subplots_adjust(left=0.15, right=0.92, bottom=0.1, top=0.9, hspace=0.5)
    plt.show()
    plt.savefig(op.join(write_dir, 'sum_acc' + '_' + key), dpi=PNG_DPI,
        facecolor='white')

# sparsity
for i_key, key in enumerate(target_studies):
    plt.figure(figsize=(8, 8))
    for i_ana in np.arange(4):
        fig_id = plt.subplot(4, 1, 1 + i_ana)
        my_ticks = mat_actual_nparc[i_SVM_PEN, i_ana + i_key * 4, :, 0][:n_items]
        plt.xticks(my_ticks.astype(np.int))
        for i_compr, compr in enumerate(compressors):
            plt.plot(
                my_ticks,
                mat_spars[i_SVM_PEN, i_ana + i_key * 4, :, i_compr][:n_items],
                label=compr)
        plt.ylabel('Sparsity (L1/L2)')
        plt.title(analysis_modes[i_ana + i_key * 4] +
                  '/SVM-' + SVM_PEN_list[i_SVM_PEN])
        max_sp = np.max(mat_spars[i_SVM_PEN, i_ana + i_key * 4, :, i_compr][:n_items])
        plt.yticks(np.linspace(0, max_sp, 11))
        # plt.ylim(0., 30.)
        plt.yticks(np.linspace(0., 30., 7))
        plt.legend(loc='lower right', fontsize=8)
    plt.xlabel('#Components')
    plt.subplots_adjust(left=0.15, right=0.92, bottom=0.1, top=0.9, hspace=0.5)
    plt.show()
    plt.savefig(op.join(write_dir, 'sum_spars' + '_' + key), dpi=PNG_DPI,
        facecolor='white')

# recall
for i_key, key in enumerate(target_studies):
    plt.figure(figsize=(8, 8))
    for i_ana in np.arange(4):
        fig_id = plt.subplot(4, 1, 1 + i_ana)
        my_ticks = mat_actual_nparc[i_SVM_PEN, i_ana, :, 0][:n_items]
        plt.xticks(my_ticks.astype(np.int))
        for i_compr, compr in enumerate(compressors):
            plt.plot(
                my_ticks,
                mat_rec[i_SVM_PEN, i_ana + i_key * 4, :, i_compr][:n_items],
                label=compr)
        plt.ylabel('Recall')
        plt.title(analysis_modes[i_ana + i_key * 4] +
                  '/SVM-' + SVM_PEN_list[i_SVM_PEN])
        max_sp = np.max(mat_rec[i_SVM_PEN, i_ana + i_key * 4, :, i_compr][:n_items])
        plt.yticks(np.linspace(0, max_sp, 11))
        # plt.ylim(0, 1.)
        plt.yticks(np.linspace(0, 1, 11))
        plt.legend(loc='lower right', fontsize=8)
    plt.xlabel('#Components')
    plt.subplots_adjust(left=0.15, right=0.92, bottom=0.1, top=0.9, hspace=0.5)
    plt.show()
    plt.savefig(op.join(write_dir, 'sum_rec' + '_' + key), dpi=PNG_DPI,
        facecolor='white')

# precision
for i_key, key in enumerate(target_studies):
    plt.figure(figsize=(8, 8))
    for i_ana in np.arange(4):
        fig_id = plt.subplot(4, 1, 1 + i_ana)
        my_ticks = mat_actual_nparc[i_SVM_PEN, i_ana, :, 0][:n_items]
        plt.xticks(my_ticks.astype(np.int))
        for i_compr, compr in enumerate(compressors):
            plt.plot(
                my_ticks,
                mat_prec[i_SVM_PEN, i_ana + i_key * 4, :, i_compr][:n_items],
                label=compr)
        plt.ylabel('Precision')

        plt.title(analysis_modes[i_ana + i_key * 4] +
                  '/SVM-' + SVM_PEN_list[i_SVM_PEN])
        max_sp = np.max(mat_prec[i_SVM_PEN, i_ana + i_key * 4, :, i_compr][:n_items])
        plt.yticks(np.linspace(0, max_sp, 11))
        # plt.ylim(0, 1.)
        plt.yticks(np.linspace(0, 1, 11))
        plt.legend(loc='lower right', fontsize=8)
    plt.xlabel('#Components')
    plt.subplots_adjust(left=0.15, right=0.92, bottom=0.1, top=0.9, hspace=0.5)
    plt.show()
    plt.savefig(op.join(write_dir, 'sum_prec' + '_' + key), dpi=PNG_DPI,
        facecolor='white')

