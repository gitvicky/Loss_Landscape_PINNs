#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 09:23:51 2021

@author: Vicky

PINN solving 1D Burgers Equations. Numerical Schemes built using FFT. 
"""
# %%
#Numerical Solver
import numpy as np
import matplotlib.pyplot as plt 
from scipy.integrate import odeint

nu = 0.01/np.pi
L = 2
N = 1000 #Discretisation Points
dx = L/N 
x = np.arange(-L/2, L/2, dx) #Define the X Domain 

#Define the discrete wavenumbers 
kappa = 2*np.pi*np.fft.fftfreq(N, d=dx)

#Initial Condition
# u0 = 1/np.cosh(x)
u0 = - np.sin(np.pi*x) +  1/np.cosh(x)

#Simulate in Fourier Freq domain. 
dt = 0.001

t= np.arange(0, 1000*dt, dt)

def rhsBurgers(u, t, kappa, nu):
    uhat = np.fft.fft(u)
    d_uhat = (1j)*kappa*uhat
    dd_uhat = -np.power(kappa, 2)*uhat
    d_u = np.fft.ifft(d_uhat)
    dd_u = np.fft.ifft(dd_uhat)
    du_dt = -u*d_u + nu*dd_u
    return du_dt.real

u = odeint(rhsBurgers, u0, t, args=(kappa, nu))

plt.figure()
plt.imshow(np.flipud(u), aspect=.8)
plt.axis('off')
plt.set_cmap('jet')

lb = np.array([np.min(x), np.min(t)])
ub = np.array([np.max(x), np.max(t)])


# %%
#Training the PINN  with various levels of Sparse Data

import os
import time
from tqdm import tqdm 
import numpy as np 
from matplotlib import pyplot as plt 
import operator
from functools import reduce 


from pyDOE import lhs
import torch
import torch.nn as nn

# %
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

# Setting the random seed. 
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

path = os.getcwd()


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
    

#Setting up a derivative function that goes through the graph and calculates via chain rule the derivative of u wrt x 
deriv = lambda u, x: torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, allow_unused=True)[0]



# %%
#Function to train the PINN
def Burgers_PINN(sparse_val):

    configuration = {"Type": 'MLP',
                "Blocks": None,
                "Layers": 5,
                "Neurons": 100,
                "Activation": 'Tanh',
                "Optimizer": 'Adam',
                "Learning Rate": 1e-3,
                "Learning Rate Scheme": "Step LR",
                "Epochs": 2,
                "N_initial": 500,
                "N_boundary": 500,  # Each Boundary
                "N_domain": 5000,
                "Domain_Split": ' None',
                "MC Integration": 'None',
                "Regularisation": 'None',
                "Normalisation Strategy": 'None',
                "Domain": 'x -> [-1, 1], t -> [0, 1]',
                "Sparseness": sparse_val,
                "Coarseness": None,
                "LBFGS": 0,
                "Note": ''}


    run_id = int(configuration['Sparseness'])

    #Setting up an instance of the MLP with the needed architecture. 
    npde_net = MLP(2, 1, configuration['Layers'], configuration['Neurons'])
    npde_net = npde_net.to(default_device)



    #Fucnction to sample collocation points across the spatio-temporal domain using a Latin Hypercube
    def LHS_Sampling(N):
        return lb + (ub-lb)*lhs(2, N)


    #Domain Loss Function - measuring the deviation from the PDE functional. 
    def pde(X):

        x = X[:, 0:1]
        t = X[:, 1:2]
        u = npde_net(torch.cat([x,t],1))

        u_x = deriv(u, x)
        u_xx = deriv(u_x, x)
        u_t = deriv(u, t)
        
        pde_loss = u_t + u*u_x - nu*u_xx

        return pde_loss.pow(2).mean()


    #Boundary Loss Function - measuring the deviation from boundary conditions for f(x_lim, y_lim, t)
    def boundary(X_left, X_right):

        u_left = npde_net(X_left)
        u_right = npde_net(X_right)
        
        bc_loss = (u_left - u_right) 

        return bc_loss.pow(2).mean()


    #Reconstruction Loss Function - measuring the deviation fromt the actual output. Used to calculate the initial loss
    def reconstruction(X, Y):
        u = npde_net(X)
        recon_loss = u-Y
        return recon_loss.pow(2).mean()


    #Samples taken from each region for optimisation purposes. 

    N_i = configuration['N_initial']
    N_b = configuration['N_boundary']
    N_f = configuration['N_domain']

    #Prepping the inputs and outputs for the testing against the actual solution. 
    X, T = np.meshgrid(x, t)
    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    u_actual = np.expand_dims(u.flatten(),1)


    # Data for Initial Input

    X_IC = np.hstack((np.expand_dims(x, -1), np.zeros(len(t)).reshape(len(t), 1)))
    u_IC = u[0].flatten()
    u_IC = np.expand_dims(u_IC, 1)

    idx = np.random.choice(X_IC.shape[0], N_i, replace=False) 
    X_i = X_IC[idx]
    u_i = u_IC[idx]

    # Data for Boundary Input

    X_left = LHS_Sampling(N_b)
    X_left[:,0:1] = min(x)

    X_right = LHS_Sampling(N_b)
    X_right[:,0:1] = max(x)

    #Data for Domain Input
    X_f = LHS_Sampling(N_f)


    if configuration['Sparseness'] != 0:
        sparseness = configuration['Sparseness'] / 100
        data_size = len(u.flatten())
        sparse_indices = np.random.randint(data_size, size=int(sparseness * data_size))

        X_sparse = X_star[sparse_indices]
        Y_sparse = u_actual[sparse_indices]


    #Converting to tensors 

    X_i = torch_tensor_grad(X_i, default_device)
    Y_i = torch_tensor_nograd(u_i, default_device)
    X_lb = torch_tensor_grad(X_left, default_device)
    X_rb = torch_tensor_grad(X_right, default_device)

    X_f = torch_tensor_grad(X_f, default_device)

    if configuration['Sparseness'] != 0:
        X_sp = torch_tensor_grad(X_sparse, default_device)
        Y_sp = torch_tensor_grad(Y_sparse, default_device)


    optimizer = torch.optim.Adam(npde_net.parameters(), lr=configuration['Learning Rate'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.9)

    it=0
    epochs = configuration['Epochs']  
    loss_list = []

    start_time = time.time()
    while it < epochs :
        optimizer.zero_grad()

        initial_loss = reconstruction(X_i, Y_i) 
        boundary_loss = boundary(X_lb, X_rb)
        domain_loss = pde(X_f)
        if configuration['Sparseness'] !=0 :
            recon_loss = reconstruction(X_sp, Y_sp)
        else:
            recon_loss = 0 

        loss = initial_loss + boundary_loss + domain_loss + recon_loss

        
        loss.backward()
        optimizer.step()
        scheduler.step()

        it += 1

        print('It: %d, Init: %.3e, Bound: %.3e, Domain: %.3e, Sparse: %.3e' % (it, initial_loss, boundary_loss, domain_loss, recon_loss))


    train_time = time.time() - start_time

    torch.save({
        'epoch': it,
        'model_state_dict': npde_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, path + '/models/' + 'Burgers_'+str(run_id)+'.pth')


    #Getting the trained output. 
    if default_device == 'cpu':
        with torch.no_grad():
            u_pred = npde_net(torch_tensor_grad(X_star, default_device)).detach().numpy()

    else : 
        with torch.no_grad():
            u_pred = npde_net(torch_tensor_grad(X_star, default_device)).cpu().detach().numpy()
            
    l2_error = np.mean((u_actual - u_pred)**2)

    print('Training Time: %d seconds, L2 Error: %.3e' % (train_time, l2_error))

    u_pred = u_pred.reshape(len(u), len(t))


# %%
# Training the Vanilla and Sparse PINNs
Burgers_PINN(0)
Burgers_PINN(1)
# %%
#Estimating the Loss Landscapes
import loss_landscape as ll

# Prep the data.
X, T = np.meshgrid(x, t)
X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
Y_star = np.expand_dims(u.flatten(),1)


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

models = ['Burgers_0.pth', 'Burgers_1.pth']
for ii in range(len(models)):
    print('starting' + str(ii))
    name = models[ii][:-4]

    # Checkpoint
    model_loc = path + '/models/' + models[ii]

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

    # Mapping the Loss Landscape

    losses, accuracies = [], []

    # xcoordinates = np.linspace(-1, 1, 101)
    # ycoordinates = np.linspace(-1, 1, 101)

    xcoordinates = np.linspace(-1, 1, 11)
    ycoordinates = np.linspace(-1, 1, 11)

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
    plt.savefig()
    

    np.save(path + '/loss_landscapes/Loss_Surface_' + name, losses)



# %%
