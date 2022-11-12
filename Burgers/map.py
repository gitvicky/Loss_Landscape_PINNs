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


models = ['Burgers_0.pth', 'Burgers_1.pth']
# Prep the data.
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# %%
nu = 0.01 / np.pi
L = 2
N = 1000  # Discretisation Points
dx = L / N
x = np.arange(-L / 2, L / 2, dx)  # Define the X Domain

# Define the discrete wavenumbers
kappa = 2 * np.pi * np.fft.fftfreq(N, d=dx)

# Initial Condition
# u0 = 1/np.cosh(x)
u0 = - np.sin(np.pi * x) + 1 / np.cosh(x)

# Simulate in Fourier Freq domain.
dt = 0.001

t = np.arange(0, 1000 * dt, dt)


def rhsBurgers(u, t, kappa, nu):
    uhat = np.fft.fft(u)
    d_uhat = (1j) * kappa * uhat
    dd_uhat = -np.power(kappa, 2) * uhat
    d_u = np.fft.ifft(d_uhat)
    dd_u = np.fft.ifft(dd_uhat)
    du_dt = -u * d_u + nu * dd_u
    return du_dt.real


u = odeint(rhsBurgers, u0, t, args=(kappa, nu))


lb = np.array([np.min(x), np.min(t)])
ub = np.array([np.max(x), np.max(t)])

# %%
X, T = np.meshgrid(x, t)
X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
Y_star = np.expand_dims(u.flatten(),1)


# %%

def swish(x):
    return torch.mul(x, torch.nn.Sigmoid()(x))


# %%
# Fully Connected Network or a Multi-Layer Perceptron as the PINN.
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


# Torch Save Models
# model_loc = '/Users/vgopakum/Documents/Code_2021/MHD_PINNs/Models/1z1kvc1w_MHD.pth'
# model = Resnet(3, 1, 100)
# if default_device == 'cpu':
#     model.load_state_dict(torch.load(model_loc, map_location='cpu'))
# else:
#     model.load_state_dict(torch.load(model_loc))

# model.to(default_device)

for ii in range(len(models)):
    print('starting' + str(ii))
    name = models[ii][:-4]

    # Checkpoint
    model_loc = '/home/vgopakum/Training/Hybrid_PINNs/Burgers/models/' + models[ii]

    model = MLP(2, 1, 5, 100)
    # model = nn.DataParallel(model)
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

    np.save('Loss_Surface_' + name, losses)

