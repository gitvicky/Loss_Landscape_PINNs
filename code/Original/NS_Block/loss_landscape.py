#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 27 14:38:06 2021

@author: vgopakum

Python Script for Mappping the Loss Landscape - Modified from Tom Goldstein's original Github repo. 

"""
 # %%
#Load the packages 

import numpy as np
from matplotlib import pyplot as plt
import torch 
import torch.nn as nn 
import copy
import time 
from tqdm import tqdm 


seed = 42
torch.manual_seed(seed)
np.random.seed(seed)


default_device = "cuda" if torch.cuda.is_available() else "cpu"

dtype=torch.float32
torch.set_default_dtype(dtype)

def torch_tensor_grad(x, device):
    if device == 'cuda':
        x = torch.cuda.FloatTensor(x)
    else:
        x = torch.FloatTensor(x)
    x.requires_grad = True
    return x 

def torch_tensor_nograd(x, device):
    if device == 'cuda':
        x = torch.cuda.FloatTensor(x)
    else:
        x = torch.FloatTensor(x)
    x.requires_grad = False
    return x 

# %% 

# Estimating plotting resolution
xmin, xmax, xnum = -1, 1, 101
ymin, ymax, ynum = -1, 1, 101

# %%
#Projection Functions 
"""
    Project a model or multiple models to a plane spaned by given directions.
"""

def nplist_to_tensor(nplist):
    """ Concatenate a list of numpy vectors into one tensor.
        Args:
            nplist: a list of numpy vectors, e.g., direction loaded from h5 file.
        Returns:
            concatnated 1D tensor
    """
    v = []
    for d in nplist:
        w = torch.tensor(d*np.float64(1.0))
        # Ignoreing the scalar values (w.dim() = 0).
        if w.dim() > 1:
            v.append(w.view(w.numel()))
        elif w.dim() == 1:
            v.append(w)
    return torch.cat(v)




def cal_angle(vec1, vec2):
    """ Calculate cosine similarities between two torch tensors or two ndarraies
        Args:
            vec1, vec2: two tensors or numpy ndarraies
    """
    if isinstance(vec1, torch.Tensor) and isinstance(vec1, torch.Tensor):
        return torch.dot(vec1, vec2)/(vec1.norm()*vec2.norm()).item()
    elif isinstance(vec1, np.ndarray) and isinstance(vec2, np.ndarray):
        return np.ndarray.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))


# %%

#Functions for model manipulation 

def get_weights(net):
    """ Extract parameters from net, and return a list of tensors"""
    return [p.data for p in net.parameters()]

def get_random_weights(weights):
    """
        Produce a random direction that is a list of random Gaussian tensors
        with the same shape as the network's weights, so one direction entry per weight.
    """
    return [torch.randn(w.size()) for w in weights]

def set_weights(net, weights, directions=None, step=None):
    """
        Overwrite the network's weights with a specified list of tensors
        or change weights along directions with a step size.
    """
    if directions is None:
        # You cannot specify a step length without a direction.
        for (p, w) in zip(net.parameters(), weights):
            p.data.copy_(w.type(type(p.data)))
    else:
        assert step is not None, 'If a direction is specified then step must be specified as well'

        if len(directions) == 2:
            dx = directions[0]
            dy = directions[1]
            changes = [d0*step[0] + d1*step[1] for (d0, d1) in zip(dx, dy)]
        else:
            changes = [d*step for d in directions[0]]

        for (p, w, d) in zip(net.parameters(), weights, changes):
            p.data = w + torch.Tensor(d).type(type(w))

def normalize_direction(direction, weights, norm='filter'):
    """
        Rescale the direction so that it has similar norm as their corresponding
        model in different levels.
        Args:
          direction: a variables of the random direction for one layer
          weights: a variable of the original model for one layer
          norm: normalization method, 'filter' | 'layer' | 'weight'
    """
    if norm == 'filter':
        # Rescale the filters (weights in group) in 'direction' so that each
        # filter has the same norm as its corresponding filter in 'weights'.
        for d, w in zip(direction, weights):
            d.mul_(w.norm()/(d.norm() + 1e-10))
    elif norm == 'layer':
        # Rescale the layer variables in the direction so that each layer has
        # the same norm as the layer variables in weights.
        direction.mul_(weights.norm()/direction.norm())
    elif norm == 'weight':
        # Rescale the entries in the direction so that each entry has the same
        # scale as the corresponding weight.
        direction.mul_(weights)
    elif norm == 'dfilter':
        # Rescale the entries in the direction so that each filter direction
        # has the unit norm.
        for d in direction:
            d.div_(d.norm() + 1e-10)
    elif norm == 'dlayer':
        # Rescale the entries in the direction so that each layer direction has
        # the unit norm.
        direction.div_(direction.norm())


def normalize_directions_for_weights(direction, weights, norm='filter', ignore='biasbn'):
    """
        The normalization scales the direction entries according to the entries of weights.
    """
    assert(len(direction) == len(weights))
    for d, w in zip(direction, weights):
        if d.dim() <= 1:
            if ignore == 'biasbn':
                d.fill_(0) # ignore directions for weights with 1 dimension
            else:
                d.copy_(w) # keep directions for weights/bias that are only 1 per node
        else:
            normalize_direction(d, w, norm)

def create_random_direction(net, dir_type='weights', ignore='biasbn', norm='filter'):
    """
        Setup a random (normalized) direction with the same dimension as
        the weights or states.
        Args:
          net: the given trained model
          dir_type: 'weights' or 'states', type of directions.
          ignore: 'biasbn', ignore biases and BN parameters.
          norm: direction normalization method, including
                'filter" | 'layer' | 'weight' | 'dlayer' | 'dfilter'
        Returns:
          direction: a random direction with the same dimension as weights or states.
    """

    # random direction
    if dir_type == 'weights':
        weights = get_weights(net) # a list of parameters.
        direction = get_random_weights(weights)
        normalize_directions_for_weights(direction, weights, norm, ignore)
        
    return direction


# %%

def get_unplotted_indices(vals, xcoordinates, ycoordinates=None):
    """
    Args:
      vals: values at (x, y), with value -1 when the value is not yet calculated.
      xcoordinates: x locations, i.e.,[-1, -0.5, 0, 0.5, 1]
      ycoordinates: y locations, i.e.,[-1, -0.5, 0, 0.5, 1]
    Returns:
      - a list of indices into vals for points that have not yet been calculated.
      - a list of corresponding coordinates, with one x/y coordinate per row.
    """

    # Create a list of indices into the vectorizes vals
    inds = np.array(range(vals.size))

    # Select the indices of the un-recorded entries, assuming un-recorded entries
    # will be smaller than zero. In case some vals (other than loss values) are
    # negative and those indexces will be selected again and calcualted over and over.
    inds = inds[vals.ravel() <= 0]

    # Make lists containing the x- and y-coodinates of the points to be plotted
    if ycoordinates is not None:
        # If the plot is 2D, then use meshgrid to enumerate all coordinates in the 2D mesh
        xcoord_mesh, ycoord_mesh = np.meshgrid(xcoordinates, ycoordinates)
        s1 = xcoord_mesh.ravel()[inds]
        s2 = ycoord_mesh.ravel()[inds]
        return inds, np.c_[s1,s2]
    else:
        return inds, xcoordinates.ravel()[inds]


def split_inds(num_inds, nproc):
    """
    Evenly slice out a set of jobs that are handled by each MPI process.
      - Assuming each job takes the same amount of time.
      - Each process handles an (approx) equal size slice of jobs.
      - If the number of processes is larger than rows to divide up, then some
        high-rank processes will receive an empty slice rows, e.g., there will be
        3, 2, 2, 2 jobs assigned to rank0, rank1, rank2, rank3 given 9 jobs with 4
        MPI processes.
    """

    chunk = num_inds // nproc
    remainder = num_inds % nproc
    splitted_idx = []
    for rank in range(0, nproc):
        # Set the starting index for this slice
        start_idx = rank * chunk + min(rank, remainder)
        # The stopping index can't go beyond the end of the array
        stop_idx = start_idx + chunk + (rank < remainder)
        splitted_idx.append(range(start_idx, stop_idx))

    return splitted_idx


def get_job_indices(vals, xcoordinates, ycoordinates, comm):
    """
    Prepare the job indices over which coordinate to calculate.
    Args:
        vals: the value matrix
        xcoordinates: x locations, i.e.,[-1, -0.5, 0, 0.5, 1]
        ycoordinates: y locations, i.e.,[-1, -0.5, 0, 0.5, 1]
        comm: MPI environment
    Returns:
        inds: indices that splitted for current rank
        coords: coordinates for current rank
        inds_nums: max number of indices for all ranks
    """

    inds, coords = get_unplotted_indices(vals, xcoordinates, ycoordinates)

    rank = 0 if comm is None else comm.Get_rank()
    nproc = 1 if comm is None else comm.Get_size()
    splitted_idx = split_inds(len(inds), nproc)

    # Split the indices over the available MPI processes
    inds = inds[splitted_idx[rank]]
    coords = coords[splitted_idx[rank]]

    # Figure out the number of jobs that each MPI process needs to calculate.
    inds_nums = [len(idx) for idx in splitted_idx]

    return inds, coords, inds_nums



