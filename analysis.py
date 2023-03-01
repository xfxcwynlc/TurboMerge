import numpy as np
import collections
import heapq
import pickle
import cc3d
import math
import tifffile
import os
'''
Analysis of algorithm's voxel sizes, each nuclei count. Develop overlapping,similar scores for each segmentation masks

'''

if __name__ == '__main__':
    fname = '031.tif'
    # mesmer:/Users/yananw/Desktop/mesmer/membrane/031.tif
    mes = tifffile.imread(f'/Users/yananw/Desktop/mesmer/membrane/{fname}')
    # cellpose:  /Users/yananw/Desktop/cellpose/membrane/031.tif
    cellpose = tifffile.imread(f'/Users/yananw/Desktop/cellpose/membrane/{fname}')
    # turbovoxel: /Users/yananw/Desktop/resultnew/031.tif
    turbovoxel = tifffile.imread(f'/Users/yananw/Desktop/resultnew/{fname}')


    meslabels,mescts = np.unique(mes,return_counts=True)
    cellposelabels,cellposects = np.unique(cellpose,return_counts=True)
    turbolabels,turbocts = np.unique(turbovoxel,return_counts=True)

    #Analysis 1, myocyte counts
    print(f'Analysis on myocytes: {fname}')
    print(f'Mesmer unique myocytes: {len(meslabels)}')
    print(f'Cellpose unique myocytes: {len(cellposelabels)}')
    print(f'Turbovoxel unique myocytes: {len(turbolabels)}')

    #Analysis 2, area overlapping? For the same voxel look at how they differs at 3 different places
    #(10,176,260)
    coord = (10,176,260)
    print(f'\nLabel at position in zyx: {coord}')
    prefix = ['mes_','cellpose_','turbo_']
    vols = [mes,cellpose,turbovoxel]
    for s,v in zip(prefix,vols):
        l = v[coord]
        print(f'\n{s}: {l}')
        temp = 'analysis/'+s+str(l)+'_'+fname
        print(f'Write file {temp}')
        tifffile.imwrite(temp,(200*(v==l)).astype(np.uint8),photometric='minisblack')

