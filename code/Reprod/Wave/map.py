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

models = ['Wave_Sparse_0.pth', 'Wave_Sparse_1.pth']

# Prep the data.
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


class WaveEquation:
    
    def __init__(self, N):
        self.N = N
        self.x0 = -1.0
        self.xf = 1.0
        self.y0 = -1.0
        self.yf = 1.0
        self.initialization()
        self.initCond()
        
        
    def initialization(self):
        k = np.arange(self.N + 1)
        self.x = np.cos(k*np.pi/self.N)
        self.y = self.x.copy()
        self.xx, self.yy = np.meshgrid(self.x, self.y)
        
        self.dt = 6/self.N**2
        self.plotgap = round((1/3)/self.dt)
        self.dt = (1/3)/self.plotgap
        
        
    def initCond(self):
        self.vv = np.exp(-40*((self.xx-0.4)**2 + self.yy**2))
        self.vvold = self.vv.copy()
        
        
    def solve_and_animate(self):
        
        u_list = []
        # fig = plt.figure()
        
        # ax = fig.add_subplot(111, projection='3d')
        
        tc = 0
        nstep = round(3*self.plotgap+1)
        wframe = None
        
        while tc < nstep:
            # if wframe:
            #     ax.collections.remove(wframe)
                
            xxx = np.arange(self.x0, self.xf+1/16, 1/16)
            yyy = np.arange(self.y0, self.yf+1/16, 1/16)
            vvv = interpolate.interp2d(self.x, self.y, self.vv, kind='cubic')
            Z = vvv(xxx, yyy)
            
            xxf, yyf = np.meshgrid(np.arange(self.x0,self.xf+1/16,1/16), np.arange(self.y0,self.yf+1/16,1/16))
                
            # wframe = ax.plot_surface(xxf, yyf, Z, cmap=cm.coolwarm, linewidth=0, 
            #         antialiased=False)
            
            # ax.set_xlim3d(self.x0, self.xf)
            # ax.set_ylim3d(self.y0, self.yf)
            # ax.set_zlim3d(-0.15, 1)
            
            # ax.set_xlabel("x")
            # ax.set_ylabel("y")
            # ax.set_zlabel("U")
            
            # ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
            # ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
            
            # fig.suptitle("Time = %1.3f" % (tc/(3*self.plotgap-1)-self.dt))
            # #plt.tight_layout()
            # ax.view_init(elev=30., azim=-110)
            # plt.pause(0.01)
                
            uxx = np.zeros((self.N+1, self.N+1))
            uyy = np.zeros((self.N+1, self.N+1))
            ii = np.arange(1, self.N)
            
            for i in range(1, self.N):
                v = self.vv[i,:]
                V = np.hstack((v, np.flipud(v[ii])))
                U = np.fft.fft(V)
                U = U.real
                
                r1 = np.arange(self.N)
                r2 = 1j*np.hstack((r1, 0, -r1[:0:-1]))*U
                W1 = np.fft.ifft(r2)
                W1 = W1.real
                s1 = np.arange(self.N+1)
                s2 = np.hstack((s1, -s1[self.N-1:0:-1]))
                s3 = -s2**2*U
                W2 = np.fft.ifft(s3)
                W2 = W2.real
                
                uxx[i,ii] = W2[ii]/(1-self.x[ii]**2) - self.x[ii]*W1[ii]/(1-self.x[ii]**2)**(3/2)
                
            for j in range(1, self.N):
                v = self.vv[:,j]
                V = np.hstack((v, np.flipud(v[ii])))
                U = np.fft.fft(V)
                U = U.real
                
                r1 = np.arange(self.N)
                r2 = 1j*np.hstack((r1, 0, -r1[:0:-1]))*U
                W1 = np.fft.ifft(r2)
                W1 = W1.real
                s1 = np.arange(self.N+1)
                s2 = np.hstack((s1, -s1[self.N-1:0:-1]))
                s3 = -s2**2*U
                W2 = np.fft.ifft(s3)
                W2 = W2.real
                
                uyy[ii,j] = W2[ii]/(1-self.y[ii]**2) - self.y[ii]*W1[ii]/(1-self.y[ii]**2)**(3/2)
                
            vvnew = 2*self.vv - self.vvold + self.dt**2*(uxx+uyy)
            self.vvold = self.vv.copy()
            self.vv = vvnew.copy()
            tc += 1
            
            u_list.append(Z)
        return np.asarray(u_list)
        
    
def main():
    simulator = WaveEquation(30)
    u_sol = simulator.solve_and_animate()
    return u_sol

    
if __name__ == "__main__":
    u_sol = main()
    
    N = 30
    dt =  6/N**2
    k = 30 + 1
    
    lb = np.asarray([-1.0, -1.0, 0]) #[x, y, t]
    ub = np.asarray([1.0, 1.0, 1])
    
    x = np.arange(-1, 1+1/16, 1/16)
    y = x.copy()
    t = np.arange(lb[2], ub[2]+dt, dt)
    
    xx, yy = np.meshgrid(x, y)
    u_ic = np.exp(-40*((xx-0.4)**2 + yy**2))
    
    U_sol = u_sol

    grid_length = len(x)
    
    df_dict = {'x': x,
               'y': y,
               't': t,
               'lower_range': lb,
               'upper_range': ub,
               'U_sol': U_sol}

u = np.asarray(u_sol)
X, Y = np.meshgrid(x, y)
XY_star = np.hstack((X.flatten()[:,None], Y.flatten()[:,None]))
T_star = np.expand_dims(np.repeat(t, len(XY_star)), 1)
X_star_tiled = np.tile(XY_star, (len(t), 1))
X_star = np.hstack((X_star_tiled, T_star))
Y_star = np.expand_dims(u.flatten(),1)

# %%

def swish(x):
    return torch.mul(x, torch.nn.Sigmoid()(x))


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
    def __init__(self, in_features, out_features, num_neurons, activation=swish):
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
        
        x_temp = self.act_func(self.layer_after_block(x_temp))
        x_temp = self.layer_output(x_temp)

        return x_temp

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

    # model = MLP(3, 3, 8, 32)
    model = Resnet(3, 1, 100)

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