#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 17:57:11 2021

@author: vgopakum

Visualising the maped out Loss Landscapes 
"""

# %%
import numpy as np
from matplotlib import pyplot as plt 

import os 
path = os.getcwd()


import sys 

# %%
def visualiser_3d(data, name):
    xcoordinates = np.linspace(-1, 1, 101)
    ycoordinates = np.linspace(-1, 1, 101)

    X,Y = np.meshgrid(xcoordinates, ycoordinates)
    log_data = np.log(data)
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, log_data, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_title(name)
    # plt.xlabel('Projection along x-axis')
    # plt.ylabel('Projection along y-axis')
    # plt.zlabel('Log Loss')



# %%
#NAVIER STOKES

loc = path + '/NS_Block/loss_maps/'

sparse_0 = np.load(loc + 'Loss_Surface_Sparse_0.npy')
sparse_1 = np.load(loc + 'Loss_Surface_Sparse_1.npy')
sparse_5 = np.load(loc + 'Loss_Surface_Sparse_5.npy')
sparse_10 = np.load(loc + 'Loss_Surface_Sparse_10.npy')
coarse_10 = np.load(loc + 'Loss_Surface_coarse_10.npy')
line_data = np.load(loc + 'Loss_Surface_LineData.npy')

visualiser_3d(sparse_0, '0')
plt.savefig('NS_LL_0.png')
visualiser_3d(sparse_1, '1')
plt.savefig('NS_LL_1.png')
visualiser_3d(sparse_5, '5')
plt.savefig('NS_LL_5.png')
visualiser_3d(sparse_10, '10')
plt.savefig('NS_LL_10.png')
visualiser_3d(coarse_10, '')
plt.savefig('NS_LL_coarse_10.png')
visualiser_3d(line_data, '')
plt.savefig('NS_LL_linedata.png')


# %%

# loc = path + '/Navier_Stokes/loss_maps/'
# coarse_10 = np.load(loc + 'Loss_Surface_coarse_10.npy')
# visualiser_3d(coarse_10, '')


# %%
    
# linedata = np.load(loc + 'Loss_Surface_LinearData.npy')
# visualiser_3d(linedata, '')


# %%
#BURGERS
loc = path + '/Burgers/'

sparse_0 = np.load(loc + 'Loss_Surface_Burgers_0.npy')
sparse_1 = np.load(loc + 'Loss_Surface_Burgers_1.npy')

visualiser_3d(sparse_0, '0')
visualiser_3d(sparse_1, '1')