#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 17:47:16 2021

@author: vgopakum

2D Wave Equation. 
Solution obtained via the Spectral Method.
Code for Numerical Solver can be found here: http://people.bu.edu/andasari/courses/numericalpython/python.html 
```
PDE:
u_tt = u_xx + u_yy on [-1,1] x [-1,1] 

Initial Distribution :
    u(x,y,t=0) = exp(-40(x-4)^2 + y^2)


Initial Velocity Condition : 
    u_t(x,y,t=0) 

Boundary Condition : 
    u=0 (at xlim or ylim)


Domain:
    x ∈ [-1,1], y ∈ [-1,1], t ∈ [0,1]

"""

# %%
import sys
import os
import time
from tqdm import tqdm 
import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt 
from matplotlib import cm
from matplotlib import animation
from matplotlib.animation import FuncAnimation, PillowWriter


from pyDOE import lhs
import torch
import torch.nn as nn


# %%

import wandb 
run_name = 'Wave Eqn - Sparse'


configuration={"Type": 'ResNet',
                "Blocks": 2,
                "Layers": 2,
                "Neurons": 64, 
                "Activation": 'swish',
                "Optimizer": 'Adam',
                "Learning Rate": 1e-3,
                "Learning Rate Scheme": "Step LR",
                "Epochs": 20000,
                "N_initial": 1000,
                "N_boundary": 1000,  #Each Boundary
                "N_domain": 20000,
                "Domain_Split": 'None', 
                "MC Integratrion": 'Yes',
                "Normalisation Strategy": 'None',
                "Domain": 'x -> [-1, 1], y -> [-1, 1], t -> [0, 1]', 
                "constants": 'D=1',
                "Sparseness": 50.0, 
                "Coarseness": None}

run = wandb.init(project='Hybrid_PINNs', name=run_name, 
            notes='Wave Equation tests on Sparse PINNs. ',
            config=configuration)

run_id = wandb.run.id



# %%

default_device = "cuda" if torch.cuda.is_available() else "cpu"
# default_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


# Setting the random seed. 
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)


# %%
#Running the Spectral solver 
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
    
        
# %%
#Specifying the Domain of Interest. 
x_range = [-1.0, 1.0]
y_range = [-1.0, 1.0]
t_range = [0.0, 1.0]
D = 1.0

lb = np.asarray([x_range[0], y_range[0], t_range[0]])
ub = np.asarray([x_range[1], y_range[1], t_range[1]])

def LHS_Sampling(N, lb, ub):
    return lb + (ub-lb)*lhs(3, N)


# %%
def swish(x):
    return torch.mul(x, torch.nn.Sigmoid()(x))
# %%
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

#Setting up a derivative function that goes through the graph and calculates via chain rule the derivative of u wrt x 
deriv = lambda u, x: torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, allow_unused=True)[0]


# %%
#Setting up an instance of the Resnet with the needed architecture. 
npde_net = Resnet(3, 1, 64)
npde_net = npde_net.to(default_device)
# npde_net = nn.DataParallel(npde_net)
wandb.watch(npde_net, log='all')



# %%
#Setting up the loss functions
def pde(X):

    x = X[:, 0:1]
    y = X[:, 1:2]
    t = X[:, 2:3]
    u = npde_net(torch.cat([x,y,t],1))

    u_x = deriv(u, x)
    u_xx = deriv(u_x, x)
    u_y = deriv(u, y)
    u_yy = deriv(u_y, y)
    u_t = deriv(u, t)
    u_tt = deriv(u_t, t)
    
    pde_loss = u_tt - (u_xx + u_yy)

    return 2*2*1*pde_loss.pow(2).mean()


#Boundary Loss Function - measuring the deviation from boundary conditions for f(x_lim, y_lim, t)
def boundary(X):

    u = npde_net(X)
    bc_loss = (u - 0)

    return 2*bc_loss.pow(2).mean()

#Initial Velocity Conditions :
def initial_velocity(X):

    x = X[:, 0:1]
    y = X[:, 1:2]
    t = X[:, 2:3]
    u = npde_net(torch.cat([x,y,t],1))

    u_t = deriv(u, t)
    initial_cond_loss = u_t - 0

    return 2*2*initial_cond_loss.pow(2).mean()


#Reconstruction Loss Function - measuring the deviation fromt the actual output. Used to calculate the initial loss
def reconstruction(X, Y):
    u = npde_net(X)
    recon_loss = u-Y
    return 2*2*recon_loss.pow(2).mean()

# %%
#Normalisation Strategies
def min_max_norm(x):
    return (x-np.min(x))/(np.max(x)-np.min(x)) 

def z_score(x):
    return (x-np.mean(x)) / np.std(x)

def identity(x):
  return x


# %%
#Training Data 
#Samples taken from each region for optimisation purposes. 
N_i = configuration['N_initial']
N_b = configuration['N_boundary']
N_f = configuration['N_domain']


# %%
#Prepping the training data
u = np.asarray(u_sol)
X, Y = np.meshgrid(x, y)
XY_star = np.hstack((X.flatten()[:,None], Y.flatten()[:,None]))
T_star = np.expand_dims(np.repeat(t, len(XY_star)), 1)
X_star_tiled = np.tile(XY_star, (len(t), 1))
X_star = np.hstack((X_star_tiled, T_star))
Y_star = np.expand_dims(u.flatten(),1)


# Data for Initial Input 
X_IC = np.hstack((XY_star, np.zeros(len(XY_star)).reshape(len(XY_star), 1)))
u_IC = u[0].flatten()
u_IC = np.expand_dims(u_IC, 1)

idx = np.random.choice(X_IC.shape[0], N_i, replace=False) 
X_i = X_IC[idx]
u_i = u_IC[idx]

# Data for Boundary Input

X_left = LHS_Sampling(N_b, lb, ub)
X_left[:,0:1] = x_range[0]

X_right = LHS_Sampling(N_b, lb, ub)
X_right[:,0:1] = x_range[1]

X_bottom = LHS_Sampling(N_b, lb, ub)
X_bottom[:,1:2] = y_range[0]

X_top = LHS_Sampling(N_b, lb, ub)
X_top[:,1:2] = y_range[1]

X_b = np.vstack((X_right, X_top, X_left, X_bottom))
np.random.shuffle(X_b) 

#Data for Domain Input
X_f = LHS_Sampling(N_f, lb, ub)

#Prepping the sparse data 
sparseness = configuration['Sparseness']/100
if sparseness != None:
    data_size = len(X_star)
    sparse_indices = np.random.randint(data_size, size=int(sparseness*data_size))
    X_sparse = X_star[sparse_indices]
    Y_sparse = Y_star[sparse_indices]
# %%
#Converting to tensors 

X_i = torch_tensor_grad(X_i, default_device)
Y_i = torch_tensor_nograd(u_i, default_device)
X_b = torch_tensor_grad(X_b, default_device)
X_f = torch_tensor_grad(X_f, default_device)
X_sparse = torch_tensor_grad(X_sparse, default_device)
Y_sparse = torch_tensor_nograd(Y_sparse, default_device)

# %%
#Training Loop
model = "untrained"

if model == "trained":
    npde_net = Resnet(3, 1, 64)
    npde_net = npde_net.to(default_device)

    optimizer = torch.optim.Adam(npde_net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.9)

    checkpoint = torch.load('runID_Wave.pth')
    npde_net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

train_time = 0

# %%
#Training Loop
if model == "untrained":

    optimizer = torch.optim.Adam(npde_net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.9)

    if configuration['Domain_Split'] == 'None':

        it=0
        epochs = configuration['Epochs']
        start_time = time.time()

        while it < epochs :
            optimizer.zero_grad()

            initial_loss = reconstruction(X_i, Y_i) + initial_velocity(X_i)
            boundary_loss = boundary(X_b)
            domain_loss = pde(X_f)
            sparse_loss = reconstruction(X_sparse, Y_sparse)

            loss = initial_loss + boundary_loss + domain_loss  + sparse_loss

            wandb.log({'Initial Loss': initial_loss, 
                    'Boundary Loss': boundary_loss,
                    'Domain Loss': domain_loss,
                    'Sparse Loss': sparse_loss,
                    'Total Loss ': loss})
            
            loss.backward()
            optimizer.step()
            scheduler.step()

            it += 1
            print('It: %d, Loss: %.3e' % (it, loss))

            if it%5000 == 0 :
                torch.save({
                    'epoch': it,
                    'model_state_dict': npde_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss
                    }, run_id+'_Wave.pth')


    train_time = time.time() - start_time

    if configuration['Domain_Split'] != 'None':
        it=0
        epochs = configuration['Epochs']

        no_domain_until = int(epochs/4)
        time_domain_split = 3

        start_time = time.time()

        while it < no_domain_until:
            optimizer.zero_grad()

            initial_loss = reconstruction(X_i, Y_i) + initial_velocity(X_i)
            boundary_loss = boundary(X_b)
            domain_loss = 0
            sparse_loss = reconstruction(X_sparse, Y_sparse)

            loss = initial_loss + boundary_loss + sparse_loss + domain_loss
            
            wandb.log({'Initial Loss': initial_loss, 
                    'Boundary Loss': boundary_loss,
                    'Domain Loss': domain_loss,
                    'Sparse Loss': sparse_loss,
                    'Total Loss ': loss})

            loss.backward()
            optimizer.step()
            scheduler.step()

            it += 1
            print('It: %d, Loss: %.3e' % (it, loss))
            
            if it%5000 == 0 :
                torch.save({
                        'epoch': it,
                        'model_state_dict': npde_net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss
                        }, run_id+'_Wave.pth')
            
            
        for ii in range(time_domain_split):
            
            lb_split = np.asarray([x_range[0], y_range[0], int(ii*t_range[1]/3)]) #Assumes that the time domain started from 0. 
            ub_split = np.asarray([x_range[1], y_range[1], int((ii+1)*t_range[1]/3)])
            X_f = LHS_Sampling(N_f, lb_split, ub_split)
            X_f_torch = torch_tensor_grad(X_f, default_device)

            iter_till = it + (epochs-no_domain_until)/3
            while it < iter_till:
                optimizer.zero_grad()

                initial_loss = reconstruction(X_i, Y_i) + initial_velocity(X_i)
                boundary_loss = boundary(X_b)
                domain_loss = pde(X_f_torch)
                sparse_loss = reconstruction(X_sparse, Y_sparse)

                loss = initial_loss + boundary_loss + sparse_loss + domain_loss
                
                wandb.log({'Initial Loss': initial_loss, 
                        'Boundary Loss': boundary_loss,
                        'Domain Loss': domain_loss,
                        'Sparse Loss': sparse_loss,
                        'Total Loss ': loss})
            
                loss.backward()
                optimizer.step()
                scheduler.step()
            
                it += 1
                print('It: %d, Loss: %.3e' % (it, loss))
            
                if it%5000 == 0 :
                    torch.save({
                            'epoch': it,
                            'model_state_dict': npde_net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss
                            }, run_id+'_Wave.pth')

        train_time = time.time() - start_time

wandb.run.summary['Training Time'] = train_time 
 
# %%

#Getting the trained output. 
if default_device == 'cpu':
    with torch.no_grad():
        u_pred = npde_net(torch_tensor_grad(X_star, default_device)).detach().numpy()

else : 
    with torch.no_grad():
        u_pred = npde_net(torch_tensor_grad(X_star, default_device)).cpu().detach().numpy()
        
l2_error = np.mean((Y_star - u_pred)**2)

print('Training Time: %d seconds, L2 Error: %.3e' % (train_time, l2_error))

u_pred = u_pred.reshape(len(u), grid_length, grid_length)

wandb.run.summary['l2 Error'] = l2_error
# %%

u_field = u_sol

fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(2,3,1)
ax.imshow(u_field[0], cmap=cm.coolwarm)
ax.title.set_text('Initial')
ax.set_ylabel('Solution')

ax = fig.add_subplot(2,3,2)
ax.imshow(u_field[int(len(u_field)/2)], cmap=cm.coolwarm)
ax.title.set_text('Middle')

ax = fig.add_subplot(2,3,3)
ax.imshow(u_field[-1], cmap=cm.coolwarm)
ax.title.set_text('Final')


u_field = u_pred

ax = fig.add_subplot(2,3,4)
ax.imshow(u_field[0], cmap=cm.coolwarm)
ax.set_ylabel('PINN')

ax = fig.add_subplot(2,3,5)
ax.imshow(u_field[int(len(u_field)/2)], cmap=cm.coolwarm)

ax = fig.add_subplot(2,3,6)
ax.imshow(u_field[-1], cmap=cm.coolwarm)


wandb.log({"Evolution Comparison Plots": plt})

# %%
# Saving the Source Code
wandb.save('Wave.py')
wandb.save(run_id+'Wave.pth')
wandb.run.finish()

# %%
