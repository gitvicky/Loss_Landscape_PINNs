#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 10:23:30 2021

@author: vgopakum

Flow past an obstacle (velocity-pressure formulation)
 Finite Difference applied in python.
http://cav2012.sg/cdohl/CFD_course/Raizan%2520FDM%2520obstacle.html
http://www.thevisualroom.com/poisson_for_pressure.html
"""
# %%

import wandb
configuration = {"Type": 'MLP',
                 "Blocks": None,
                 "Layers": 5,
                 "Neurons": 100,
                 "Activation": 'Tanh',
                 "Optimizer": 'Adam',
                 "Learning Rate": 5e-3,
                 "Learning Rate Scheme": "Step LR",
                 "Epochs": 20000,
                 "N_initial": 1000,
                 "N_boundary": 500,  # Each Boundary
                 "N_block": 500,
                 "N_domain": 50000,
                 "Domain_Split": ' None',
                 "MC Integration": 'None',
                 "Regularisation": 'None',
                 "Normalisation Strategy": 'None',
                 "Domain": 'x -> [0,20], y -> [0,10], t -> [0, 1]',
                 "Numerical Spatial Discretisation": '201x101',
                 "constants": 'rho=1.0, nu=0.04, Re=50',
                 "Sparseness": None,
                 "Coarseness": None,
                 "Location": True,
                 "LBFGS": 0,
                 "Note": 'Removed the points within a block for domain_loss'}

run = wandb.init(project='NavStokes Laminar Block',
                 notes='Only Sparse Recon Error',
                 config=configuration)

run_id = wandb.run.id

wandb.save('NS_Block.py')

# %%

import os 

path = os.getcwd()

import sys
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import math as mth


# %%

def presPoisson(p, dx, dy, rho, botb, dpth, lftb, wdth, ny, nx):
    pn = np.empty_like(p)
    p = np.zeros((ny, nx))

    # Term in square brackets
    b[1:-1, 1:-1] = rho * (
            1 / dt * ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx) + (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) - \
            ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx)) ** 2 - \
            2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) * (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx)) - \
            ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) ** 2)

    for q in range(nit):
        pn = p.copy()
        p[1:-1, 1:-1] = ((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy ** 2 + (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx ** 2) / \
                        (2 * (dx ** 2 + dy ** 2)) - \
                        dx ** 2 * dy ** 2 / (2 * (dx ** 2 + dy ** 2)) * b[1:-1, 1:-1]

        # Apply the Neumann boundary condition as recommended above on all sides
        p[-1, :] = p[-2, :] - rho * nu / dy * (-2 * v[-2, :] + v[-3, :])  ## at y = 2
        p[0, :] = p[1, :] - rho * nu / dy * (-2 * v[1, :] + v[2, :])  ## at y = 0

        p[:, 0] = p[:, 1] - rho * nu / dx * (-2 * u[:, 1] + u[:, 2])  ## at x = 0
        p[:, -1] = p[:, -2] - rho * nu / dx * (-2 * u[:, -2] + u[:, -3])  ## at x = 2

        # We apply the same concept for boundary conditions at the top and bottom surfaces of the obstacles.
        # At bottom surface:
        p[botb, lftb:(lftb + wdth + 1)] = p[botb - 1, lftb:(lftb + wdth + 1)] - rho * nu / dy * (
                -2 * v[botb - 1, lftb:(lftb + wdth + 1)] + v[botb - 2, lftb:(lftb + wdth + 1)])

        # At top surface:
        p[(botb + dpth), lftb:(lftb + wdth + 1)] = p[(botb + dpth + 1), lftb:(lftb + wdth + 1)] - rho * nu / dy * (
                -2 * v[(botb + dpth + 1), lftb:(lftb + wdth + 1)] + v[(botb + dpth + 2),
                                                                    lftb:(lftb + wdth + 1)])  # at y = 0

        # Likewise for the right and left surfaces of the obstacles
        # At the left surface:
        p[botb:(botb + dpth + 1), lftb] = p[botb:(botb + dpth + 1), lftb - 1] - rho * nu / dx * (
                -2 * u[botb:(botb + dpth + 1), lftb - 1] + u[botb:(botb + dpth + 1), lftb - 2])  # at x = 2

        # At the right surface:
        p[botb:(botb + dpth + 1), (lftb + wdth)] = p[botb:(botb + dpth + 1), (lftb + wdth + 1)] - rho * nu / dx * (
                -2 * u[botb:(botb + dpth + 1), (lftb + wdth + 1)] + u[botb:(botb + dpth + 1),
                                                                    (lftb + wdth + 2)])  # at x = 0

        # Pressure values inside obstacle should be zero
        # since there is no pressure flux in and out of the obstacle
        p[(botb + 1):(botb + dpth), (lftb + 1):(lftb + wdth)] = 0

    return p


def cavityFlow(nt, u, v, dt, dx, dy, p, rho, nu, botb, dpth, lftb, wdth, X, Y, u_start):
    un = np.empty_like(u)
    vn = np.empty_like(v)
    b = np.zeros((ny, nx))

    u_list = []
    v_list = []
    p_list = []

    # --------------------------------------------
    # Initialise u values as initial condition
    # --------------------------------------------
    u[:, 0] = u_start

    # -------------------------------------
    # Start iteration through timesteps
    # -------------------------------------
    for n in range(nt):
        un = u.copy()
        vn = v.copy()

        u_list.append(un)
        v_list.append(vn)
        p_list.append(p)

        p = presPoisson(p, dx, dy, rho, botb, dpth, lftb, wdth, ny, nx)

        # ===================================
        # to locate position of maximum pressure and the maximum value itself, and the corresponding U and V
        # for debugging purposes
        # print(np.where(p == p.max()))
        # print(p.max())
        # print ("--- time: " + str(n))
        # print("P:" + str(p[40,68]))
        # print("U:" + str(u[40,68]))
        # print("V:" + str(v[40,68]))
        # ===================================

        u[1:-1, 1:-1] = un[1:-1, 1:-1] - \
                        un[1:-1, 1:-1] * dt / dx * (un[1:-1, 1:-1] - un[1:-1, 0:-2]) - \
                        vn[1:-1, 1:-1] * dt / dy * (un[1:-1, 1:-1] - un[0:-2, 1:-1]) - \
                        dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2]) + \
                        nu * (dt / dx ** 2 * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) + \
                              dt / dy ** 2 * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1]))

        v[1:-1, 1:-1] = vn[1:-1, 1:-1] - \
                        un[1:-1, 1:-1] * dt / dx * (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) - \
                        vn[1:-1, 1:-1] * dt / dy * (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) - \
                        dt / (2 * rho * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1]) + \
                        nu * (dt / dx ** 2 * (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) + \
                              (dt / dy ** 2 * (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))

        # -------------------------------------------------------------
        # Apply boundary conditions to the inlet, exit points, as well
        # as the top and bottom boundary conditions.
        # -------------------------------------------------------------
        # Prescribing inlet boundary condition at the inlet itself.
        # i.e. at every time step, a constant u-direction speed enters the pipe
        # Ref: Eq. 2.53 "Essential Computational Fluid Dynamics", Zikanov (2010)
        u[:, 0] = u_start

        # Near the exit, only an artificial boundary condition can be set since the
        # flow is artificially cut off, and therefore not possible to predict
        # what will occur at the exit, and how it can conversely affect flow
        # within the domain. Set zero gradient at the exit in the direction of x for each
        # time step
        # Ref: Eq. 2.54 "Essential Computational Fluid Dynamics", Zikanov (2010)
        u[:, -1] = u[:, -2]

        # Bottom and top surface of pipe has zero tangential velocity - no slip boundary condition
        u[0, :] = 0
        u[-1, :] = 0

        # Also set the vertical velocity at the inlet and exit to be zero, i.e. force laminar flow
        v[:, -1] = 0  # at exit
        v[:, 0] = 0  # at inlet

        # likewise vertical velocity at each of the bottom and top surface is also zero
        v[0, :] = 0  # bottom surface
        v[-1, :] = 0  # top surface

        # -------------------------------------------------------------
        # Apply boundary conditions to the obstacle
        # -------------------------------------------------------------
        # zero velocity everywhere at the obstacle
        u[botb:(botb + dpth + 1), lftb:(lftb + wdth + 1)] = 0
        v[botb:(botb + dpth + 1), lftb:(lftb + wdth + 1)] = 0

    u_sol = np.asarray(u_list)
    v_sol = np.asarray(v_list)
    p_sol = np.asarray(p_list)

    return u_sol, v_sol, p_sol


# %%

nx = 201  # x-points
ny = 101  # y-points
nit = 50
c = 1  # phase propagation speed
x_span = 20.0
y_span = 10.0
dx = x_span / (nx - 1)  # size of x-grid
dy = y_span / (ny - 1)  # size of y-grid
x = np.linspace(0, x_span, nx)  # last point included, so exactly nx points
y = np.linspace(0, y_span, ny)  # last point included, so exactly ny points
X, Y = np.meshgrid(x, y)  # makes 2-dimensional mesh grid

botb = 40  # bottom boundary of obstacle
dpth = 20  # obstacle depth

lftb = 70  # left boundary of obstacle
wdth = 5  # obstacle width

Re = 50  # range from 10s to 100s
nt = 1000  # timesteps

#characteristic units 
L = 2.0
U = 1.0

u_start = 1  # initial velocity at the start
rho = 1.0  # density
nu = ((dy * dpth) * u_start) / Re  # viscosity (UL/Re, Re = UL/nu, original value: 0.1)
dt = 0.001  # timesteps
t = np.arange(0, 1, dt)

qres = 3  # quiver plot resolution

v = np.zeros((ny, nx))
u = np.ones((ny, nx))  # for u-velocity I initialise to 1 everywhere

p = np.zeros((ny, nx))
b = np.zeros((ny, nx))

u_sol, v_sol, p_sol = cavityFlow(nt, u, v, dt, dx, dy, p, rho, nu, botb, dpth, lftb, wdth, X, Y, u_start)

# %%
time_ii = -1
# Plot the last figure on screen
fig = plt.figure(figsize=(100, 50), dpi=25)
plt.contourf(X, Y, p_sol[time_ii], alpha=0.5)
plt.tick_params(axis='both', which='major', labelsize=40)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=50)
plt.contour(X, Y, p_sol[time_ii])
plt.quiver(X[::qres, ::qres], Y[::qres, ::qres], u_sol[time_ii][::qres, ::qres],
           v_sol[time_ii][::qres, ::qres])  ##plotting velocity
plt.broken_barh([(x[lftb + 1], x[lftb + wdth - 2] - x[lftb + 1])], (y[botb + 1], y[botb + dpth - 2] - y[botb + 1]),
                facecolors='grey', alpha=0.8)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('time_step = ' + str(nt) + ' nu = ' + str(nu), fontsize=40)

# %%

# #Dimensionless conversions 
# x = x/L
# y = y/L
# t = t/(L/U)
# u_sol = u_sol/U
# v_sol = v_sol/U
# p_sol = p_sol/(rho*U**2)

# %%

import os
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
# %matplotlib inline
import operator
from functools import reduce 

from pyDOE import lhs
import torch
import torch.nn as nn
# import torch.autograd.profiler as profiler
from torch.utils.data import Dataset, DataLoader

# Setting the random seed.
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

default_device = "cuda" if torch.cuda.is_available() else "cpu"

dtype = torch.float32
torch.set_default_dtype(dtype)


def torch_tensor_grad(x, device):
    if device == 'cuda':
        x = torch.cuda.FloatTensor(x).to(device)
    else:
        x = torch.FloatTensor(x).to(device)
    x.requires_grad = True
    return x


def torch_tensor_nograd(x, device):
    if device == 'cuda':
        x = torch.cuda.FloatTensor(x).to(device)
    else:
        x = torch.FloatTensor(x).to(device)
    x.requires_grad = False
    return x


# %%
def swish(x):
    return torch.mul(x, torch.nn.Sigmoid()(x))


class Resnet(nn.Module):
    def __init__(self, in_features, out_features, num_neurons, activation=torch.tanh):
        super(Resnet, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_neurons = num_neurons

        self.act_func = activation

        self.block1_layer1 = nn.Linear(self.in_features, self.num_neurons)
        self.block1_layer2 = nn.Linear(self.num_neurons, self.num_neurons)
        self.block1 = [self.block1_layer1, self.block1_layer2]

        self.block2_layer1 = nn.Linear(self.in_features + self.num_neurons, self.num_neurons)
        self.block2_layer2 = nn.Linear(self.num_neurons, self.num_neurons)
        self.block2 = [self.block2_layer1, self.block2_layer2]

        # self.block3_layer1 = nn.Linear(self.in_features + self.num_neurons, self.num_neurons)
        # self.block3_layer2 = nn.Linear(self.num_neurons, self.num_neurons)
        # self.block3 = [self.block3_layer1, self.block3_layer2]

        # self.block4_layer1 = nn.Linear(self.in_features + self.num_neurons, self.num_neurons)
        # self.block4_layer2 = nn.Linear(self.num_neurons, self.num_neurons)
        # self.block4 = [self.block4_layer1, self.block4_layer2]

        self.layer_after_block = nn.Linear(self.num_neurons + self.in_features, self.num_neurons)
        self.layer_output = nn.Linear(self.num_neurons, self.out_features)

    def forward(self, x):

        x_temp = x

        for dense in self.block1:
            x_temp = self.act_func(dense(x_temp))
        x_temp = torch.cat([x_temp, x], dim=-1)

        for dense in self.block2:
            x_temp = self.act_func(dense(x_temp))
        x_temp = torch.cat([x_temp, x], dim=-1)

        # for dense in self.block3:
        #     x_temp = self.act_func(dense(x_temp))
        # x_temp = torch.cat([x_temp, x], dim=-1)

        # for dense in self.block4:
        #     x_temp = self.act_func(dense(x_temp))
        # x_temp = torch.cat([x_temp, x], dim=-1)

        x_temp = self.act_func(self.layer_after_block(x_temp))
        x_temp = self.layer_output(x_temp)

        return x_temp

    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))
        return c 


# Fully Connected Network or a Multi-Layer Perceptron
class MLP(nn.Module):
    def __init__(self, in_features, out_features, num_layers, num_neurons, activation=torch.tanh):
        super(MLP, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_layers = num_layers
        self.num_neurons = num_neurons

        self.act_func = activation

        self.layers = nn.ModuleList()

        self.layer_input = nn.Linear(self.in_features, self.num_neurons)

        for ii in range(self.num_layers - 1):
            self.layers.append(nn.Linear(self.num_neurons, self.num_neurons))
        self.layer_output = nn.Linear(self.num_neurons, self.out_features)

    def forward(self, x):

        x_temp = self.act_func(self.layer_input(x))
        for dense in self.layers:
            x_temp = self.act_func(dense(x_temp))
        x_temp = self.layer_output(x_temp)
        return x_temp

    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))
        return c 


class FourierNet(nn.Module):
    def __init__(self, in_features, out_features, num_layers, num_neurons, activation=torch.sin, fourier_features=64):
        super(FourierNet, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_neurons = num_neurons
        self.num_layers = num_layers
        self.fourier_features = fourier_features

        self.act_func = activation

        self.frequency_matrix = nn.Parameter(torch.randn(self.in_features, self.fourier_features))

        self.layer_1 = nn.Linear(self.fourier_features * 2, self.num_neurons)

        self.layers = nn.ModuleList()
        for ii in range(self.num_layers - 1):
            self.layers.append(nn.Linear(self.num_neurons, self.num_neurons))
        self.layer_output = nn.Linear(self.num_neurons, self.out_features)

        self.torch_pi = torch.Tensor([np.pi]).to(
            default_device)  # default_device is always mentioned in the Main function in the problem setup only.

    def forward(self, x):
        x_proj = 2 * self.torch_pi * torch.matmul(x, self.frequency_matrix)
        x_temp = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

        x_temp = self.act_func(self.layer_1(x_temp))

        for dense in self.layers:
            x_temp = self.act_func(dense(x_temp))
        x_temp = self.layer_output(x_temp)
        return x_temp
    
    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))
        return c 

class MultiProngedPINN(nn.Module):
    def __init__(self, in_features, out_features, num_layers_in, num_layers_out, num_neurons, activation=torch.tanh):
        super(MultiProngedPINN, self).__init__()

        self.num_layers_in = num_layers_in
        self.num_layers_out = num_layers_out
        self.num_neurons = num_neurons

        self.act_func = activation

        self.layers_x = nn.ModuleList()
        self.layers_y = nn.ModuleList()
        self.layers_t = nn.ModuleList()
        self.layers_out = nn.ModuleList()

        self.layer_in_x = nn.Linear(1, self.num_neurons)
        self.layer_in_y = nn.Linear(1, self.num_neurons)
        self.layer_in_t = nn.Linear(1, self.num_neurons)
        
        self.layers_concat = nn.Linear(int(num_neurons*in_features), self.num_neurons)

        for ii in range(self.num_layers_in - 1):
            self.layers_x.append(nn.Linear(self.num_neurons, self.num_neurons))
        
        for ii in range(self.num_layers_in - 1):
            self.layers_y.append(nn.Linear(self.num_neurons, self.num_neurons))

        for ii in range(self.num_layers_in - 1):
            self.layers_t.append(nn.Linear(self.num_neurons, self.num_neurons))

        for ii in range(self.num_layers_out - 1):
            self.layers_out.append(nn.Linear(self.num_neurons, self.num_neurons))
        self.layer_output = nn.Linear(self.num_neurons, out_features)

    def forward(self, X):
        x = X[:, 0:1]
        y = X[:, 1:2]
        t = X[:, 2:3]

        x = self.act_func(self.layer_in_x(x))        
        y = self.act_func(self.layer_in_y(y))
        t = self.act_func(self.layer_in_t(t))


        for dense in self.layers_x:
            x = self.act_func(dense(x))

        for dense in self.layers_y:
            y = self.act_func(dense(y))

        for dense in self.layers_t:
            t = self.act_func(dense(t))
            
        X = torch.cat([x,y,t],1)
            
        X = self.layers_concat(X)
        
        for dense in self.layers_out:
            X = self.act_func(dense(X))
        X =self.layer_output(X)
        return X 

    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))
        return c

class Bi_MultiProngedPINN(nn.Module):
    def __init__(self, in_features, out_features, num_layers_in, num_layers_out, num_layers_middle, num_neurons, activation=torch.tanh):
        super(Bi_MultiProngedPINN, self).__init__()

        self.num_layers_in = num_layers_in
        self.num_layers_out = num_layers_out
        self.num_layers_middle = num_layers_middle
        self.num_neurons = num_neurons

        self.act_func = activation

        self.layers_x = nn.ModuleList()
        self.layers_y = nn.ModuleList()
        self.layers_t = nn.ModuleList()
        self.layers = nn.ModuleList()

        self.layer_in_x = nn.Linear(1, self.num_neurons)
        self.layer_in_y = nn.Linear(1, self.num_neurons)
        self.layer_in_t = nn.Linear(1, self.num_neurons)
        
        self.layer_out_x = nn.Linear(1, self.num_neurons)
        self.layer_out_y = nn.Linear(1, self.num_neurons)
        self.layer_out_t = nn.Linear(1, self.num_neurons)
        
        
        self.layers_concat = nn.Linear(int(num_neurons*in_features), self.num_neurons)

        for ii in range(self.num_layers_in - 1):
            self.layers_x.append(nn.Linear(self.num_neurons, self.num_neurons))
        
        for ii in range(self.num_layers_in - 1):
            self.layers_y.append(nn.Linear(self.num_neurons, self.num_neurons))

        for ii in range(self.num_layers_in - 1):
            self.layers_t.append(nn.Linear(self.num_neurons, self.num_neurons))

        for ii in range(self.num_layers_middle - 1):
            self.layers.append(nn.Linear(self.num_neurons, self.num_neurons))
            
        self.layer_output = nn.Linear(self.num_neurons*in_features, out_features)

    def forward(self, X):
        x = X[:, 0:1]
        y = X[:, 1:2]
        t = X[:, 2:3]

        x = self.act_func(self.layer_in_x(x))        
        y = self.act_func(self.layer_in_y(y))
        t = self.act_func(self.layer_in_t(t))


        for dense in self.layers_x:
            x = self.act_func(dense(x))

        for dense in self.layers_y:
            y = self.act_func(dense(y))

        for dense in self.layers_t:
            t = self.act_func(dense(t))
            
        X = torch.cat([x,y,t],1)
            
        X = self.layers_concat(X)
        
        for dense in self.layers:
            X = self.act_func(dense(X))
            
        x = X[:, 0:1]
        y = X[:, 1:2]
        t = X[:, 2:3]
        
        x = self.act_func(self.layer_out_x(x))        
        y = self.act_func(self.layer_out_y(y))
        t = self.act_func(self.layer_out_t(t))
        
        X = torch.cat([x,y,t],1)

        X =self.layer_output(X)
        return X 
    
    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))   
            
        return c 
            
# %%

# Setting up a derivative function that goes through the graph and calculates via chain rule the derivative of u wrt x
deriv = lambda u, x: torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, allow_unused=True)[0]
# %%
# Setting up an instance of the Resnet with the needed architecture.
# npde_net = FourierNet(3, 3, 5, 100)
npde_net = MLP(3, 3, configuration['Layers'], configuration['Neurons'])
# npde_net = Resnet(3, 3, 64)
# npde_net = MultiProngedPINN(3, 3, configuration['Layers'], configuration['Layers'], configuration['Neurons'])
# npde_net = Bi_MultiProngedPINN(3, 3, configuration['Layers'], configuration['Layers'], configuration['Layers'], configuration['Neurons'])

# npde_net = nn.DataParallel(npde_net)
npde_net = npde_net.to(default_device)
wandb.watch(npde_net, log='all')
wandb.run.summary['Params'] = npde_net.count_params()

# Specifying the Domain of Interest.
x_range = [0.0, 20.0]
y_range = [0.0, 10.0]
t_range = [0.0, 1.0]

block_pos = {'left': 7.0,
             'right': 7.5,
             'bottom': 4.0,
             'top': 6.0}

lb = np.asarray([x_range[0], y_range[0], t_range[0]])
ub = np.asarray([x_range[1], y_range[1], t_range[1]])


def LHS_Sampling(N, lb, ub):
    return lb + (ub - lb) * lhs(3, N)


nu = 0.04
rho = 1.0
Re = 50


# %%
# Domain Loss Function - measuring the deviation from the PDE functional.

def pde(X):
    x = X[:, 0:1]
    y = X[:, 1:2]
    t = X[:, 2:3]
    out = npde_net(torch.cat([x, y, t], 1))

    u = out[:, 0:1]
    v = out[:, 1:2]
    p = out[:, 2:3]

    u_x = deriv(u, x)
    v_x = deriv(v, x)
    p_x = deriv(p, x)

    u_xx = deriv(u_x, x)
    v_xx = deriv(v_x, x)
    p_xx = deriv(p_x, x)

    u_y = deriv(u, y)
    v_y = deriv(v, y)
    p_y = deriv(p, y)

    u_yy = deriv(u_y, y)
    v_yy = deriv(v_y, y)
    p_yy = deriv(p_y, y)

    u_t = deriv(u, t)
    v_t = deriv(v, t)

    u_loss = u_t + u * u_x + v * u_y + (1 / rho) * p_x - nu * (u_xx + u_yy)
    v_loss = v_t + u * v_x + v * v_y + (1 / rho) * p_y - nu * (v_xx + v_yy)
    p_loss = p_xx + p_yy + rho * (u_x ** 2 + 2 * u_y * v_x + v_y ** 2)

    # u_loss = u_t + u * u_x + v * u_y + p_x - (1/Re) * (u_xx + u_yy) #non-dim
    # v_loss = v_t + u * v_x + v * v_y + p_y - (1/Re) * (v_xx + v_yy) #non-dim
    
    pde_loss = u_loss + v_loss + p_loss 

    return pde_loss.pow(2).mean()



def boundary_y(X):
    x = X[:, 0:1]
    y = X[:, 1:2]
    t = X[:, 2:3]
    out = npde_net(torch.cat([x, y, t], 1))

    u = out[:, 0:1]
    v = out[:, 1:2]
    p = out[:, 2:3]

    p_y = deriv(p, y)
    v_y = deriv(v, y)
    v_yy = deriv(v_y, y)

    bc_loss = p_y - rho * nu * v_yy + u + v
    # bc_loss = p_y - (1/Re) * v_yy + u/L + v/L # non-dim


    return bc_loss.pow(2).mean()


def boundary_inlet(X):
    x = X[:, 0:1]
    y = X[:, 1:2]
    t = X[:, 2:3]
    out = npde_net(torch.cat([x, y, t], 1))

    u = out[:, 0:1]
    v = out[:, 1:2]
    p = out[:, 2:3]

    p_x = deriv(p, x)
    u_x = deriv(u, x)
    u_xx = deriv(u_x, x)

    bc_loss = p_x - rho * nu * u_xx + (u-1) + v
    # bc_loss = p_x - (1/Re) * u_xx + (u-1)/L + v/L #non-dim


    return bc_loss.pow(2).mean()


def boundary_outlet(X):
    x = X[:, 0:1]
    y = X[:, 1:2]
    t = X[:, 2:3]
    out = npde_net(torch.cat([x, y, t], 1))

    u = out[:, 0:1]
    v = out[:, 1:2]
    p = out[:, 2:3]

    p_x = deriv(p, x)
    u_x = deriv(u, x)
    u_xx = deriv(u_x, x)

    bc_loss = p_x - rho * nu * u_xx + u_x + v
    # bc_loss = p_x - (1/Re) * u_xx + u_x*(U/L) + v/L #non-dim

    return bc_loss.pow(2).mean()





# Boundary Loss Function - measuring the deviation from boundary conditions for f(x_lim, y_lim, t)
def block(X):
    x = X[:, 0:1]
    y = X[:, 1:2]
    t = X[:, 2:3]
    out = npde_net(torch.cat([x, y, t], 1))

    # u = out[:, 0:1]
    # v = out[:, 1:2]
    # p = out[:, 2:3]

    bc_loss = out

    return bc_loss.pow(2).mean()


def boundary_block_perp(X):
    x = X[:, 0:1]
    y = X[:, 1:2]
    t = X[:, 2:3]
    out = npde_net(torch.cat([x, y, t], 1))

    u = out[:, 0:1]
    v = out[:, 1:2]
    p = out[:, 2:3]

    p_x = deriv(p, x)
    u_x = deriv(u, x)
    u_xx = deriv(u_x, x)

    bc_loss = p_x - rho * nu * u_xx 
    # bc_loss = p_x - (1/Re) * u_xx #non-dim


    return bc_loss.pow(2).mean()


def boundary_block_parallel(X):
    x = X[:, 0:1]
    y = X[:, 1:2]
    t = X[:, 2:3]
    out = npde_net(torch.cat([x, y, t], 1))

    u = out[:, 0:1]
    v = out[:, 1:2]
    p = out[:, 2:3]

    p_y = deriv(p, y)
    v_y = deriv(v, y)
    v_yy = deriv(v_y, y)

    bc_loss = p_y - rho * nu * v_yy
    # bc_loss = p_y - (1/Re) * v_yy #non-dim

    return bc_loss.pow(2).mean()



# Reconstruction Loss Function - measuring the deviation fromt the actual output. Used to calculate the initial loss
def reconstruction(X, Y):
    u = npde_net(X)
    recon_loss = (u - Y)
    return recon_loss.pow(2).mean()


# %%
# Normalisation Strategies

# Unit Gaussian Normalisation by way of Z-score.
class UnitGaussianNormalizer(object):
    def __init__(self, mean, std, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps  # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:, sample_idx] + self.eps  # T*batch*n
                mean = self.mean[:, sample_idx]

        # x is in shape of batch*n or T*batch*n
        x = (x * std) + mean
        return x


# normalization, scaling by range
class RangeNormalizer(object): #Min and Max for each grid cell
    def __init__(self, Min, Max, low=0.0, high=1.0):
        super(RangeNormalizer, self).__init__()
        
        mymin = Min
        mymax = Max

        self.a = (high - low) / (mymax - mymin)
        self.b = -self.a * mymax + high

    def encode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = self.a * x + self.b
        x = x.view(s)
        return x

    def decode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = (x - self.b) / self.a
        x = x.view(s)
        return x


class Min_Max(object):
    def __init__(self, Min, Max, low=0.0, high=1.0):
        self.a = (high - low) / (Max- Min)
        self.b = -self.a * Max + high

    def encode(self, x):
        x = self.a * x + self.b
        return x

    def decode(self, x):
        x = (x - self.b) / self.a
        return x

# %%
# Samples taken from each region for optimisation purposes.
N_i = configuration['N_initial']
N_b = configuration['N_boundary']
N_bl = configuration['N_block']
N_f = configuration['N_domain']

X, Y = np.meshgrid(x, y)
XY_star = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))
T_star = np.expand_dims(np.repeat(t, len(XY_star)), 1)
X_star_tiled = np.tile(XY_star, (len(t), 1))

X_star = np.hstack((X_star_tiled, T_star))
Y_star = np.hstack((np.expand_dims(u_sol.flatten(), 1),
                    np.expand_dims(v_sol.flatten(), 1),
                    np.expand_dims(p_sol.flatten(), 1)))

def data_sampling():
    # Data for Initial Input
    X_IC = LHS_Sampling(N_i, lb, ub)
    X_IC[:, 2:3] = 0
    u_IC = np.expand_dims(np.ones(N_i), 1)
    v_IC = np.expand_dims(np.zeros(N_i), 1)
    p_IC = np.expand_dims(np.zeros(N_i), 1)
    Y_IC = np.hstack((u_IC, v_IC, p_IC))
    X_i = X_IC
    Y_i = Y_IC

    # Data for Boundary Input

    X_left = LHS_Sampling(N_b, lb, ub)
    X_left[:, 0:1] = x_range[0]

    X_right = LHS_Sampling(N_b, lb, ub)
    X_right[:, 0:1] = x_range[1]

    X_bottom = LHS_Sampling(N_b, lb, ub)
    X_bottom[:, 1:2] = y_range[0]

    X_top = LHS_Sampling(N_b, lb, ub)
    X_top[:, 1:2] = y_range[1]

    X_inlet = X_left
    X_outlet = X_right

    # X_x_b = np.vstack((X_left, X_right))
    X_y_b = np.vstack((X_bottom, X_top))

    # np.random.shuffle(X_x_b)
    # np.random.shuffle(X_y_b)

    # Block Boundaries
    lb_block = np.asarray([block_pos['left'], block_pos['bottom'], t_range[0]])
    ub_block = np.asarray([block_pos['right'], block_pos['top'], t_range[1]])

    X_left_block = LHS_Sampling(N_bl, lb_block, ub_block)
    X_left_block[:, 0:1] = block_pos['left']

    X_right_block = LHS_Sampling(N_bl, lb_block, ub_block)
    X_right_block[:, 0:1] = block_pos['right']

    X_bottom_block = LHS_Sampling(N_bl, lb_block, ub_block)
    X_bottom_block[:, 1:2] = block_pos['bottom']

    X_top_block = LHS_Sampling(N_bl, lb_block, ub_block)
    X_top_block[:, 1:2] = block_pos['top']

    X_x_block = np.vstack((X_left_block, X_right_block))
    X_y_block = np.vstack((X_bottom_block, X_top_block))

    np.random.shuffle(X_x_block)
    np.random.shuffle(X_y_block)

    # Within the Block
    X_block = LHS_Sampling(N_bl, lb_block, ub_block)

    # Data for Domain Input
    X_f = LHS_Sampling(N_f, lb, ub)
    within_block_idx = np.where(np.logical_and(np.logical_and(X_f[:,1]>=block_pos['bottom'], X_f[:,1]<=block_pos['top']), np.logical_and(X_f[:,0]>=block_pos['left'], X_f[:,0]<=block_pos['right'])))
    X_f = np.delete(X_f, within_block_idx[0], axis=0) #Removing the points within the block. 

    return X_i, Y_i, X_inlet, X_outlet, X_y_b, X_x_block, X_y_block, X_block, X_f

# %%
# Loading some sparse Simulation Data
if configuration['Sparseness'] != None:
    sparseness = configuration['Sparseness'] / 100
    data_size = len(u_sol.flatten())
    sparse_indices = np.random.randint(data_size, size=int(sparseness * data_size))

    X_sparse = X_star[sparse_indices]
    Y_sparse = Y_star[sparse_indices]

    # u_sparse = np.expand_dims(u_sol.flatten(), 1)[sparse_indices]
    # v_sparse = np.expand_dims(v_sol.flatten(), 1)[sparse_indices]
    # p_sparse = np.expand_dims(p_sol.flatten(), 1)[sparse_indices]

    # Y_sparse = np.hstack((u_sparse, v_sparse, p_sparse))

# Loading Coarse Simulation Data
if configuration['Coarseness'] != None:
    coarseness = configuration['Coarseness']
    t_coarse = t[::coarseness]
    u_coarse = u_sol[::coarseness]
    v_coarse = v_sol[::coarseness]
    p_coarse = p_sol[::coarseness]

    t_star_coarse = np.expand_dims(np.repeat(t_coarse, len(XY_star)), 1)
    X_star_tiled_coarse = np.tile(XY_star, (len(t_coarse), 1))
    X_coarse = np.hstack((X_star_tiled_coarse, t_star_coarse))
    Y_coarse = np.hstack((np.expand_dims(u_coarse.flatten(), 1), np.expand_dims(v_coarse.flatten(), 1),
                          np.expand_dims(p_coarse.flatten(), 1)))

#Location Specific Data
if configuration['Location'] != None:
    yt, ty = np.meshgrid(y, t[::10])
    YT_star = np.hstack((yt.flatten()[:, None], ty.flatten()[:, None]))

    X_loc_1 = np.insert(YT_star, 0, np.ones(len(YT_star))*x[55], axis=1)
    Y_loc_1 = np.hstack((np.expand_dims(u_sol[::10, :, 55].flatten(), 1), np.expand_dims(v_sol[::10, :, 55].flatten(),1), np.expand_dims(p_sol[::10, :, 55].flatten(), 1)))

    X_loc_2 = np.insert(YT_star, 0, np.ones(len(YT_star))*x[90], axis=1)
    Y_loc_2 = np.hstack((np.expand_dims(u_sol[::10, :, 90].flatten(), 1), np.expand_dims(v_sol[::10, :, 90].flatten(),1), np.expand_dims(p_sol[::10, :, 90].flatten(), 1)))

    X_loc = np.vstack((X_loc_1, X_loc_2))
    Y_loc = np.vstack((Y_loc_1, Y_loc_2))

    # np.random.shuffle(X_loc)
    # np.random.shuffle(Y_loc)





X_i, Y_i, X_inlet, X_outlet, X_y_b, X_x_block, X_y_block, X_block, X_f = data_sampling()


# %%
# Normalisation
# Min_X = np.min(X_star, 0)
# Max_X = np.max(X_star, 0)
# input_norm = Min_Max(torch_tensor_grad(Min_X, default_device), torch_tensor_grad(Max_X, default_device))


# Min_Y = np.min(Y_star, 0)
# Max_Y = np.max(Y_star, 0)
# output_norm = Min_Max(torch_tensor_grad(Min_Y, default_device), torch_tensor_grad(Max_Y, default_device))

# Converting to tensors

X_i_torch = torch_tensor_grad(X_i, default_device)
Y_i_torch = torch_tensor_grad(Y_i, default_device)
X_inlet_torch = torch_tensor_grad(X_inlet, default_device)
X_outlet_torch = torch_tensor_grad(X_outlet, default_device)
X_y_b_torch = torch_tensor_grad(X_y_b, default_device)
X_x_block_torch = torch_tensor_grad(X_x_block, default_device)
X_y_block_torch = torch_tensor_grad(X_y_block, default_device)
X_block_torch = torch_tensor_grad(X_block, default_device)
X_f_torch = torch_tensor_grad(X_f, default_device)


# X_sparse_torch = torch_tensor_grad(X_sparse, default_device)
# Y_sparse_torch = torch_tensor_grad(Y_sparse, default_device)

# X_coarse_torch = torch_tensor_grad(X_coarse, default_device)
# Y_coarse_torch = torch_tensor_grad(Y_coarse, default_device)

X_loc_torch = torch_tensor_grad(X_loc, default_device)
Y_loc_torch = torch_tensor_grad(Y_loc, default_device)


# X_i_torch.requires_grad = True
# X_inlet_torch.requires_grad = True
# X_outlet_torch.requires_grad = True
# X_y_b_torch.requires_grad = True
# X_x_block_torch.requires_grad = True
# X_y_block_torch.requires_grad=True 
# X_block_torch.requires_grad = True
# X_f_torch.requires_grad = True 
# X_sparse_torch.requires_grad = True


# %%
# Training Loop
train_time = 0

optimizer = torch.optim.Adam(npde_net.parameters(), lr=configuration['Learning Rate'])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100000, gamma=0.9)

it = 0
epochs = configuration['Epochs'] 

a = torch_tensor_nograd([1.0], default_device)
b = torch_tensor_nograd([1.0], default_device)
bl = torch_tensor_nograd([1.0], default_device)
r = torch_tensor_nograd([1.0], default_device)

Lambda = 1.0

a_list = []
b_list = []
bl_list = []
r_list = []

#Returning the grad estimations required to calculate thte coefficients for the loss entities to address for the gradient stiffness as described in equation 7 of NSFNet paper. 
def coefficient_NSF(loss, params): 
  grads = torch.autograd.grad(loss, params, grad_outputs=torch.ones_like(loss), retain_graph=True, allow_unused=True)[0]
  return torch.abs(nn.utils.parameters_to_vector(grads)).mean()

def coefficient_PINNs_fail_num(loss, params):
  grads = torch.autograd.grad(loss, params, grad_outputs=torch.ones_like(loss), retain_graph=True, allow_unused=True)[0]
  return torch.max(torch.abs((nn.utils.parameters_to_vector(grads))))

def coefficient_PINNs_fail_den(loss, params):
  grads = torch.autograd.grad(loss, params, grad_outputs=torch.ones_like(loss), retain_graph=True, allow_unused=True)[0]
  return torch.abs(nn.utils.parameters_to_vector(grads)).mean()


# no_domain_until = int(epochs/4)
# time_domain_split = 3

start_time = time.time()

while it < epochs:
    optimizer.zero_grad()

    # X_i, Y_i, X_x_b, X_y_b, X_x_block, X_y_block, X_block, X_f = data_sampling()

    # X_i_torch = torch_tensor_grad(X_i, default_device)
    # Y_i_torch = torch_tensor_grad(Y_i, default_device)
    # X_x_b_torch = torch_tensor_grad(X_x_b, default_device)
    # X_y_b_torch = torch_tensor_grad(X_y_b, default_device)
    # X_x_block_torch = torch_tensor_grad(X_x_block, default_device)
    # X_y_block_torch = torch_tensor_grad(X_y_block, default_device)
    # X_block_torch = torch_tensor_grad(X_block, default_device)
    # X_f_torch = torch_tensor_grad(X_f, default_device)

    # X_i_torch = input_norm.encode(X_i_torch)
    # Y_i_torch = output_norm.encode(Y_i_torch)
    # X_inlet_torch = input_norm.encode(X_inlet_torch)
    # X_outlet_torch = input_norm.encode(X_outlet_torch)
    # X_y_b_torch = input_norm.encode(X_y_b_torch)
    # X_x_block_torch = input_norm.encode(X_x_block_torch)
    # X_y_block_torch = input_norm.encode(X_y_block_torch)
    # X_block_torch = input_norm.encode(X_block_torch)
    # X_f_torch = input_norm.encode(X_f_torch)

    # X_sparse_torch = input_norm.encode(X_sparse_torch)
    # Y_sparse_torch = output_norm.encode(Y_sparse_torch)

    # if it < 5000:
    #     initial_loss = reconstruction(X_i_torch, Y_i_torch)
    #     boundary_loss = boundary_inlet(X_inlet_torch) + boundary_outlet(X_outlet_torch) + boundary_y(X_y_b_torch) + boundary_block_perp(X_y_block_torch) + boundary_block_parallel(X_x_block_torch)
    #     domain_loss = 0
    #     block_loss = block(X_block_torch)
    #     recon_loss = reconstruction(X_sparse_torch, Y_sparse_torch)


    # else:
    #     initial_loss = reconstruction(X_i_torch, Y_i_torch)
    #     boundary_loss = boundary_inlet(X_inlet_torch) + boundary_outlet(X_outlet_torch) + boundary_y(X_y_b_torch) + boundary_block_perp(X_y_block_torch) + boundary_block_parallel(X_x_block_torch)
    #     domain_loss = pde(X_f_torch)
    #     block_loss = block(X_block_torch)
    #     recon_loss = reconstruction(X_sparse_torch, Y_sparse_torch)



    initial_loss = reconstruction(X_i_torch, Y_i_torch)
    boundary_loss = boundary_inlet(X_inlet_torch) + boundary_outlet(X_outlet_torch) + boundary_y(X_y_b_torch) + boundary_block_perp(X_y_block_torch) + boundary_block_parallel(X_x_block_torch)
    domain_loss = pde(X_f_torch)
    block_loss = block(X_block_torch)
    # recon_loss = reconstruction(X_sparse_torch, Y_sparse_torch)
    # recon_loss = reconstruction(X_coarse_torch, Y_coarse_torch)
    recon_loss = reconstruction(X_loc_torch, Y_loc_torch)

    # recon_loss = 0

    loss = a*initial_loss + b*boundary_loss + bl*block_loss + domain_loss + r*recon_loss

    #NSFNet
    # coefficient = coefficient_NSF
    # domain_grad = coefficient(domain_loss, npde_net.parameters())
    # initial_grad = coefficient(initial_loss, npde_net.parameters())
    # bound_grad = coefficient(boundary_loss, npde_net.parameters())
    # block_grad = coefficient(block_loss, npde_net.parameters())
    # # recon_grad = coefficient(recon_loss, npde_net.parameters())

    #PINNfail
    # domain_grad = coefficient_PINNs_fail_num(domain_loss, npde_net.parameters())
    # initial_grad = coefficient_PINNs_fail_num(a*initial_loss, npde_net.parameters())
    # bound_grad = coefficient_PINNs_fail_den(b*boundary_loss, npde_net.parameters())
    # block_grad = coefficient_PINNs_fail_den(bl*block_loss, npde_net.parameters())
    # recon_grad = coefficient_PINNs_fail_den(r*recon_loss, npde_net.parameters())

    # a_tild = domain_grad / initial_grad
    # b_tild = domain_grad / bound_grad
    # bl_tild = domain_grad / block_grad
    # # r_tild = domain_grad / recon_grad 

    it += 1
    print('It: %d, Loss: %.3e' % (it, loss))

    wandb.log({'Initial Loss': initial_loss,
               'Boundary Loss': boundary_loss,
               'Domain Loss': domain_loss,
               'Block Loss': block_loss,
               'Recon Loss': recon_loss,
               'Total Loss ': loss
            #    'a_coeff': a,
            #    'b_coeff': b,
            #    'd_coeff': torch.tensor([1.0]),
            #    'bl_coeff': bl,
            #    'r_coeff': r
               })
    


    loss.backward()
    optimizer.step()


    scheduler.step()


    # a = (1-Lambda)*a + Lambda*a_tild
    # b = (1-Lambda)*b + Lambda*b_tild
    # bl = (1-Lambda)*bl + Lambda*bl_tild
    # # r = (1-Lambda)*r + Lambda*r_tild

    a_list.append(a)
    b_list.append(b)
    bl_list.append(bl)
    r_list.append(r)


    if it % 10000 == 0:
        torch.save({
            'epoch': it,
            'model_state_dict': npde_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }, path + '/models/' + run_id + '_NS_Block.pth')
        test_loss = []
Y_pred = []

with torch.no_grad():
    for ii in tqdm(range(0, len(X_star), int(101 * 201))):
        X_test = torch_tensor_nograd(X_star[ii:ii + int(101 * 201)], default_device)
        Y_test = torch_tensor_nograd(Y_star[ii:ii + int(101 * 201)], default_device)

        # X_test = input_norm.encode(X_test)
        # Y_test = output_norm.encode(Y_test)

        pred = npde_net(X_test)
        # pred = output_norm.decode(pred)
        Y_pred.append(pred.cpu().numpy())
        test_loss.append((pred - Y_test).pow(2).mean().cpu().detach().numpy())

# %%

    Y_pred = np.asarray(Y_pred)
    l2_error = np.mean(np.asarray((test_loss)))
    Y_pred = Y_pred.reshape(len(t), len(y), len(x), 3)
    u_pred = Y_pred[..., 0]
    v_pred = Y_pred[..., 1]
    p_pred = Y_pred[..., 2]


    # %%

    # Actual Solution
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(2, 3, 1)
    ax.contourf(X, Y, p_sol[10], alpha=0.5)
    ax.contour(X, Y, p_sol[10])
    ax.quiver(X[::qres, ::qres], Y[::qres, ::qres], u_sol[10][::qres, ::qres], v_sol[10][::qres, ::qres])  ##plotting velocity
    ax.broken_barh([(x[lftb + 1], x[lftb + wdth - 2] - x[lftb + 1])], (y[botb + 1], y[botb + dpth - 2] - y[botb + 1]),
                facecolors='grey', alpha=0.8)
    ax.title.set_text('Early')
    ax.set_ylabel('Solution')

    ax = fig.add_subplot(2, 3, 2)
    ax.contourf(X, Y, p_sol[500], alpha=0.5)
    ax.contour(X, Y, p_sol[500])
    ax.quiver(X[::qres, ::qres], Y[::qres, ::qres], u_sol[500][::qres, ::qres],
            v_sol[500][::qres, ::qres])  ##plotting velocity
    ax.broken_barh([(x[lftb + 1], x[lftb + wdth - 2] - x[lftb + 1])], (y[botb + 1], y[botb + dpth - 2] - y[botb + 1]),
                facecolors='grey', alpha=0.8)
    ax.title.set_text('Middle')

    ax = fig.add_subplot(2, 3, 3)
    ax.contourf(X, Y, p_sol[-1], alpha=0.5)
    ax.contour(X, Y, p_sol[-1])
    ax.quiver(X[::qres, ::qres], Y[::qres, ::qres], u_sol[-1][::qres, ::qres],
            v_sol[-1][::qres, ::qres])  ##plotting velocity
    ax.broken_barh([(x[lftb + 1], x[lftb + wdth - 2] - x[lftb + 1])], (y[botb + 1], y[botb + dpth - 2] - y[botb + 1]),
                facecolors='grey', alpha=0.8)
    ax.title.set_text('Final')

    # Predicted Solution
    ax = fig.add_subplot(2, 3, 4)
    ax.contourf(X, Y, p_pred[1], alpha=0.5)
    ax.contour(X, Y, p_pred[1])
    ax.quiver(X[::qres, ::qres], Y[::qres, ::qres], u_pred[1][::qres, ::qres],
            v_pred[1][::qres, ::qres])  ##plotting velocity
    ax.broken_barh([(x[lftb + 1], x[lftb + wdth - 2] - x[lftb + 1])], (y[botb + 1], y[botb + dpth - 2] - y[botb + 1]),
                facecolors='grey', alpha=0.8)
    ax.set_ylabel('PINN')

    ax = fig.add_subplot(2, 3, 5)
    ax.contourf(X, Y, p_pred[500], alpha=0.5)
    ax.contour(X, Y, p_pred[500])
    ax.quiver(X[::qres, ::qres], Y[::qres, ::qres], u_pred[500][::qres, ::qres],
            v_pred[500][::qres, ::qres])  ##plotting velocity
    ax.broken_barh([(x[lftb + 1], x[lftb + wdth - 2] - x[lftb + 1])], (y[botb + 1], y[botb + dpth - 2] - y[botb + 1]),
                facecolors='grey', alpha=0.8)

    ax = fig.add_subplot(2, 3, 6)
    ax.contourf(X, Y, p_pred[-1], alpha=0.5)
    ax.contour(X, Y, p_pred[-1])
    ax.quiver(X[::qres, ::qres], Y[::qres, ::qres], u_pred[-1][::qres, ::qres],
            v_pred[-1][::qres, ::qres])  ##plotting velocity
    ax.broken_barh([(x[lftb + 1], x[lftb + wdth - 2] - x[lftb + 1])], (y[botb + 1], y[botb + dpth - 2] - y[botb + 1]),
                facecolors='grey', alpha=0.8)

    wandb.log({"img": [wandb.Image(plt, caption=str(ii))]})

train_time = time.time() - start_time

# %%
if a.requires_grad==True:
    a_list = torch.stack(a_list).cpu().detach().numpy()
    b_list = torch.stack(b_list).cpu().detach().numpy()
    bl_list = torch.stack(bl_list).cpu().detach().numpy()
    r_list = torch.stack(r_list).cpu().detach().numpy()


else:
    a_list = torch.stack(a_list).cpu().numpy()
    b_list = torch.stack(b_list).cpu().numpy()
    bl_list = torch.stack(bl_list).cpu().numpy()
    r_list = torch.stack(r_list).cpu().numpy()

fig = plt.figure()
ax = fig.add_subplot(2,2,1)
ax.plot(a_list)
ax.title.set_text('a_coeff')
ax.set_ylabel('Value')
ax.set_xticks([])

ax = fig.add_subplot(2,2,2)
ax.plot(b_list)
ax.title.set_text('b_coeff')
ax.set_xticks([])

ax = fig.add_subplot(2,2,3)
ax.plot(bl_list)
ax.title.set_text('bl_coeff')
ax.set_ylabel('Value')

# ax = fig.add_subplot(2,2,4)
# ax.plot(r_list)
# ax.title.set_text('r_coeff')

# wandb.log({"img": [wandb.Image(plt, caption='Evolution of Loss Coefficients')]})


    # %%
# class Dataset(Dataset):

#     def __init__(self, X_star, Y_star):
#         self.X_star = X_star
#         self.Y_star = Y_star

#         self.len = len(X_star)

#     def __getitem__(self, index):
#         return self.X_star, self.Y_star

#     def __len__(self):
#         return self.len

# data_loader = DataLoader(dataset=Dataset(X_star, Y_star),
#                         batch_size=int(len(x)*len(y)),
#                         shuffle=False)

# # %%
# test_loss = []
# Y_pred = []
# # X_star_grid = X_star.reshape(len(t), len(x), len(y))

# for X_test, Y_test in tqdm(data_loader):
#     with torch.no_grad():
#         X_test, Y_test = torch_tensor_nograd(X_test, default_device), torch_tensor_nograd(Y_test, default_device)
#         pred = npde_net(X_test)
#         Y_pred.append(pred)
#         test_loss.append((pred-Y_test).pow(2).mean())


# %%
test_loss = []
Y_pred = []

with torch.no_grad():
    for ii in tqdm(range(0, len(X_star), int(101 * 201))):
        X_test = torch_tensor_nograd(X_star[ii:ii + int(101 * 201)], default_device)
        Y_test = torch_tensor_nograd(Y_star[ii:ii + int(101 * 201)], default_device)

        # X_test = input_norm.encode(X_test)
        # Y_test = output_norm.encode(Y_test)

        pred = npde_net(X_test)
        # pred = output_norm.decode(pred)
        Y_pred.append(pred.cpu().numpy())
        test_loss.append((pred - Y_test).pow(2).mean().cpu().detach().numpy())

# %%

Y_pred = np.asarray(Y_pred)
l2_error = np.mean(np.asarray((test_loss)))
Y_pred = Y_pred.reshape(len(t), len(y), len(x), 3)

wandb.run.summary['Training Time'] = train_time
wandb.run.summary['l2 Error'] = l2_error
# %%

u_pred = Y_pred[..., 0]
v_pred = Y_pred[..., 1]
p_pred = Y_pred[..., 2]

print('Training Time: %d seconds, L2 Error: %.3e' % (train_time, l2_error))

# %%

# Actual Solution
fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(2, 3, 1)
ax.contourf(X, Y, p_sol[10], alpha=0.5)
ax.contour(X, Y, p_sol[10])
ax.quiver(X[::qres, ::qres], Y[::qres, ::qres], u_sol[10][::qres, ::qres], v_sol[10][::qres, ::qres])  ##plotting velocity
ax.broken_barh([(x[lftb + 1], x[lftb + wdth - 2] - x[lftb + 1])], (y[botb + 1], y[botb + dpth - 2] - y[botb + 1]),
               facecolors='grey', alpha=0.8)
ax.title.set_text('Early')
ax.set_ylabel('Solution')

ax = fig.add_subplot(2, 3, 2)
ax.contourf(X, Y, p_sol[500], alpha=0.5)
ax.contour(X, Y, p_sol[500])
ax.quiver(X[::qres, ::qres], Y[::qres, ::qres], u_sol[500][::qres, ::qres],
          v_sol[500][::qres, ::qres])  ##plotting velocity
ax.broken_barh([(x[lftb + 1], x[lftb + wdth - 2] - x[lftb + 1])], (y[botb + 1], y[botb + dpth - 2] - y[botb + 1]),
               facecolors='grey', alpha=0.8)
ax.title.set_text('Middle')

ax = fig.add_subplot(2, 3, 3)
ax.contourf(X, Y, p_sol[-1], alpha=0.5)
ax.contour(X, Y, p_sol[-1])
ax.quiver(X[::qres, ::qres], Y[::qres, ::qres], u_sol[-1][::qres, ::qres],
          v_sol[-1][::qres, ::qres])  ##plotting velocity
ax.broken_barh([(x[lftb + 1], x[lftb + wdth - 2] - x[lftb + 1])], (y[botb + 1], y[botb + dpth - 2] - y[botb + 1]),
               facecolors='grey', alpha=0.8)
ax.title.set_text('Final')

# Predicted Solution
ax = fig.add_subplot(2, 3, 4)
ax.contourf(X, Y, p_pred[1], alpha=0.5)
ax.contour(X, Y, p_pred[1])
ax.quiver(X[::qres, ::qres], Y[::qres, ::qres], u_pred[1][::qres, ::qres],
          v_pred[1][::qres, ::qres])  ##plotting velocity
ax.broken_barh([(x[lftb + 1], x[lftb + wdth - 2] - x[lftb + 1])], (y[botb + 1], y[botb + dpth - 2] - y[botb + 1]),
               facecolors='grey', alpha=0.8)
ax.set_ylabel('PINN')

ax = fig.add_subplot(2, 3, 5)
ax.contourf(X, Y, p_pred[500], alpha=0.5)
ax.contour(X, Y, p_pred[500])
ax.quiver(X[::qres, ::qres], Y[::qres, ::qres], u_pred[500][::qres, ::qres],
          v_pred[500][::qres, ::qres])  ##plotting velocity
ax.broken_barh([(x[lftb + 1], x[lftb + wdth - 2] - x[lftb + 1])], (y[botb + 1], y[botb + dpth - 2] - y[botb + 1]),
               facecolors='grey', alpha=0.8)

ax = fig.add_subplot(2, 3, 6)
ax.contourf(X, Y, p_pred[-1], alpha=0.5)
ax.contour(X, Y, p_pred[-1])
ax.quiver(X[::qres, ::qres], Y[::qres, ::qres], u_pred[-1][::qres, ::qres],
          v_pred[-1][::qres, ::qres])  ##plotting velocity
ax.broken_barh([(x[lftb + 1], x[lftb + wdth - 2] - x[lftb + 1])], (y[botb + 1], y[botb + dpth - 2] - y[botb + 1]),
               facecolors='grey', alpha=0.8)

wandb.log({"img": [wandb.Image(plt, caption='Evolution Comparison Plots')]})

# %%
# Saving the trained models
wandb.save(path + '/models/' + run_id + '_NS_Block.pth')

wandb.run.finish()

sys.exit()
