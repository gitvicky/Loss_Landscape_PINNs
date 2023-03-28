# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 27 14:40:39 2021

@author: vgopakum
"""
# %%
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
import time
import loss_landscape as ll
import os
import operator
from functools import reduce

path = os.getcwd()
results = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))) + '/results'

models = ['sparse_0.pth', 'sparse_1.pth', 'coarse_10.pth']

# Prep the data.

# %%
import sys
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import math as mth



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
X, Y = np.meshgrid(x, y)
XY_star = np.hstack((X.flatten()[:,None], Y.flatten()[:,None]))
T_star = np.expand_dims(np.repeat(t, len(XY_star)), 1)
X_star_tiled = np.tile(XY_star, (len(t), 1))

X_star = np.hstack((X_star_tiled, T_star))


u = np.expand_dims(u_sol.flatten(), 1)
v = np.expand_dims(v_sol.flatten(), 1)
p = np.expand_dims(p_sol.flatten(), 1)

Y_star = np.hstack((u, v, p))


# %%

def swish(x):
    return torch.mul(x, torch.nn.Sigmoid()(x))


# %%

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


# Setting up a derivative function that goes through the graph and calculates via chain rule the derivative of u wrt x
deriv = lambda u, x: torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, allow_unused=True)[0]


# %%
# Evaluation Function

def eval_loss(net, criterion, X, Y, use_cuda=False):
    if use_cuda:
        net.cuda()
    net.eval()

    with torch.no_grad():
        Y_tild = net(X)
        loss = criterion(Y_tild, Y)

        return loss.pow(2).mean()


# %%
# Load the model and extract the parameters - trained models

print(models)

for ii in range(len(models)):
    print('starting' + str(ii))

    name = models[ii][:-4]

    # Checkpoint
    model_loc = path + '/models/' + models[ii]

    model = MLP(3, 3, 5, 100)
    # model = Resnet(3, 1, 100)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    if ll.default_device == 'cpu':
        checkpoint = torch.load(model_loc, map_location='cpu')
    else:
        checkpoint = torch.load(model_loc)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    net = ll.copy.deepcopy(model)
    w = ll.get_weights(net)

    # Creating random directions (which are hypothised to be orthogonal to each other)
    x_direction = ll.create_random_direction(net)
    y_direction = ll.create_random_direction(net)

    d = [x_direction, y_direction]

    # calculate the consine similarity of the two directions
    if len(d) == 2:
        similarity = ll.cal_angle(ll.nplist_to_tensor(d[0]), ll.nplist_to_tensor(d[1]))
        print('cosine similarity between x-axis and y-axis: %f' % similarity)

    # %%
    # Mapping the Loss Landscape

    losses, accuracies = [], []

    xcoordinates = np.linspace(-1, 1, 101)
    ycoordinates = np.linspace(-1, 1, 101)

    shape = xcoordinates.shape if ycoordinates is None else (len(xcoordinates), len(ycoordinates))
    losses = -np.ones(shape=shape)
    accuracies = -np.ones(shape=shape)
    loss_key = losses
    acc_key = accuracies

    inds, coords, inds_nums = ll.get_job_indices(losses, xcoordinates, ycoordinates, comm=None)

    start_time = time.time()
    total_sync = 0.0

    criterion = nn.MSELoss()
    # criterion = pde
    # criterion = total_loss

    print('Starting calculations')
    # Loop over all uncalculated loss values
    for count, ind in tqdm(enumerate(inds)):
        # Get the coordinates of the loss value being calculated
        coord = coords[count]

        # Load the weights corresponding to those coordinates into the net
        ll.set_weights(net, w, d, coord)

        # Record the time to compute the loss value
        loss_start = time.time()
        loss = eval_loss(net, criterion, ll.torch_tensor_grad(X_star, ll.default_device),
                             ll.torch_tensor_grad(Y_star, ll.default_device))

        loss_compute_time = time.time() - loss_start

        # Record the result in the local array
        losses.ravel()[ind] = loss

    total_time = time.time() - start_time
    print('Total Time Taken : ' + str(total_time))
    # %%
    # Plotting the loss surface
    X, Y = np.meshgrid(xcoordinates, ycoordinates)
    loss_data_log = np.log(losses)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, loss_data_log, rstride=1, cstride=1, cmap='viridis', edgecolor='none')

    ax.set_title('Logarithm: Surface Plot of Loss Landscape')

    plt.savefig(results + '/loss_landscape_'+name+'.png')
    np.save(results + '/Loss_Surface_' + name, losses)