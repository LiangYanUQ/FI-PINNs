import torch
import torch.nn as nn
import os
import numpy as np
import sys 
import shutil 
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.append('FI_PINNs')

from utils.freeze_weights import freeze_by_idxs


device = torch.device("cuda:0")

class DNN(nn.Module):
    """This class carrys out DNN"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_hiddens):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_hiddens = num_hiddens
        self.nn = [
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.Tanh()
        ]
        for _ in range(self.num_hiddens):
            self.nn += self.block()
        self.nn.append(nn.Linear(self.hidden_dim, self.output_dim))
        self.nn = nn.Sequential(*self.nn)
        # self.nn.apply(self.init_weights)

    def block(self):
        """This block implements a hidden block"""
        return [nn.Linear(self.hidden_dim, self.hidden_dim), nn.Tanh()]
    
    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_normal_(m.weight, 0.1)
            m.bias.data.fill_(0.001)

    def forward(self, x):
        return self.nn(x)


class PinnOnePeak:
    """This script carrys out unbounded pinn pdes"""
    def __init__(self, X_b_train, u_b, img_save_path) -> None:
        self.img_save_path = img_save_path
        self.loss_func = nn.MSELoss()
        self.iter = 0
        self.net = DNN(2, 20, 1, 7).to(device)

        self.u_b = torch.tensor(u_b, dtype = torch.float32).to(device)
        self.x_b = torch.tensor(X_b_train[:, 0].reshape(-1, 1),
                                dtype=torch.float32,
                                requires_grad=True).to(device)
        self.y_b = torch.tensor(X_b_train[:, 1].reshape(-1, 1),
                                dtype=torch.float32,
                                requires_grad=True).to(device)

        u_true = lambda x, y: np.exp(-1000*((x-0.5)**2 + (y - 0.5)**2)) 
        x = np.linspace(-1,1,100)
        y = np.linspace(-1,1,100)
        X, Y = np.meshgrid(x, y)
        self.points = np.array([X.flatten(), Y.flatten()]).T
        self.true_u = u_true(X.flatten(), Y.flatten()).reshape(X.shape)
        self.X, self.Y = X, Y
        self.optim_adam = torch.optim.Adam(self.net.parameters(), lr = 1e-4)


        

    def net_u(self, x, y):
        u = self.net( torch.hstack((x, y)) )
        return u

    def net_f(self, x, y):
        u = self.net_u(x, y)
        
        u_y = torch.autograd.grad(
            u, y, 
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True)[0]
        
        u_x = torch.autograd.grad(
            u, x, 
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True)[0]
        
        u_xx = torch.autograd.grad(
            u_x, x, 
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True)[0]
        u_yy = torch.autograd.grad(
            u_y, y, 
            grad_outputs=torch.ones_like(u_y),
            retain_graph=True,
            create_graph=True)[0]

        f =  -u_yy - u_xx - self.source_function(x, y)

        return f
    
    def source_function(self, x, y):
        temp = -1000*((x - 0.5)**2 + (y - 0.5)**2)
        return 4000 * (temp + 1) * torch.exp(temp)


    def closure(self):
        self.optimizer.zero_grad()
        
        # u & f predictions:
        u_b_prediction = self.net_u(self.x_b, self.y_b)
        f_prediction = self.net_f(self.x_f, self.y_f)

        # losses:
        u_b_loss = self.loss_func(u_b_prediction, self.u_b)
        f_loss = self.loss_func(f_prediction, torch.zeros_like(f_prediction).to(device))
        ls = f_loss + u_b_loss

        # derivative with respect to net's weights:
        ls.backward()
        self.error.append(self.calculate_error())

        # increase iteration count:
        self.iter += 1

        # print report:
        if not self.iter % 10:
            print('Epoch: {0:}, Loss: {1:.4f}'.format(self.iter, ls))

        return ls

    def train(self, X_f_train, adam_iters, i = 0):
        self.update(X_f_train)
        self.net.train()
        self.error = []
        batch_sz = self.x_f.shape[0]
        n_batches =  self.x_f.shape[0] // batch_sz
        if i >= 1:
            freeze_by_idxs(self.net, [0, 1, 2])
        # self.optimizer.step(self.closure)
        for i in range(adam_iters):
            for j in range(n_batches):
                x_f_batch = self.x_f[j*batch_sz:(j*batch_sz + batch_sz),]
                y_f_batch = self.y_f[j*batch_sz:(j*batch_sz + batch_sz),]
                self.optim_adam.zero_grad()
                u_b_prediction = self.net_u(self.x_b, self.y_b)
                f_prediction = self.net_f(x_f_batch, y_f_batch)

                # losses:
                u_b_loss = self.loss_func(u_b_prediction, self.u_b)
                f_loss = self.loss_func(f_prediction, torch.zeros_like(f_prediction).to(device))
                ls = f_loss + u_b_loss
                ls.backward()
                
                self.optim_adam.step()
            # scheduler.step(ls)
            if not i % 100:
                self.error.append(self.calculate_error())
                print('current epoch: %d, loss: %.7f'%(i, ls.item()))
        print('---------------------------------------------------')
        # self.optimizer.step(self.closure)
    
         

    def update(self, X_f_train):
        self.x_f = torch.tensor(X_f_train[:, 0].reshape(-1, 1),
                                dtype=torch.float32,
                                requires_grad=True).to(device)
        self.y_f = torch.tensor(X_f_train[:, 1].reshape(-1, 1),
                                dtype=torch.float32,
                                requires_grad=True).to(device)

    def predict(self, points):
        x = torch.tensor(points[:, 0:1], requires_grad = True).float().to(device)
        y = torch.tensor(points[:, 1:2], requires_grad = True).float().to(device)

        self.net.eval()
        u = self.net_u(x, y)
        f = self.net_f(x, y)
        u = u.to('cpu').detach().numpy()
        f = f.to('cpu').detach().numpy()
        return u, f

    def plot_error(self, add_points = None, prefix = None):
        """ plot the solution on new data """
        u_predict, f_predict = self.predict(self.points)
    
        u_predict = u_predict.reshape(self.true_u.shape)
        f_predict = f_predict.reshape(self.true_u.shape)

        fig = plt.figure(figsize=(15,10))
        fig.suptitle('Initial points:' + str(len(self.y_f)))
        ax1 = fig.add_subplot(221)
        im1 = ax1.contourf(self.X, self.Y, abs(f_predict), cmap = "winter")
        if add_points is not None:
            fig.suptitle('Initial points: ' + str(len(self.x_f)) +  ' ' + 'add points: ' + str(len(add_points)))
            ax1.scatter(add_points[:,0], add_points[:,1], marker = 'o', edgecolors = 'red', facecolors = 'white')
        ax1.set_title("Equation error")
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('right', size='2%', pad=0.08)
        fig.colorbar(im1, cax=cax, orientation='vertical')
    
        ax2 = fig.add_subplot(222)
        im2 = ax2.contourf(self.X, self.Y, abs(u_predict - self.true_u), cmap = "winter")
        ax2.set_title("Solution error")
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes('right', size='2%', pad=0.08)
        fig.colorbar(im2, cax=cax, orientation='vertical')
        
        ax3 = fig.add_subplot(223)
        ax3.plot(self.error, label = "l_2 error")
        ax3.set_xlabel('Epoches * 100')
        ax3.legend()
        plt.savefig(os.path.join(self.img_save_path, prefix + '.png'))
        plt.close()
        # plt.show()
    
    def calculate_error(self):
        u_predict, _ = self.predict(self.points)
        error = np.linalg.norm(u_predict.squeeze() - self.true_u.flatten())/np.linalg.norm(self.true_u.flatten())
        return error
    
    def absolute_error(self):
        u_predict, _ = self.predict(self.points)
        error = np.linalg.norm(u_predict.squeeze() - self.true_u.flatten())
        return error

