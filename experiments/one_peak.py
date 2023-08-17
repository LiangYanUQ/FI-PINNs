import sys
sys.path.append('FI_PINNs')

import torch 
import numpy as np
from functools import partial
import os
import copy 
import pickle 
from model.pinn_one_peak_torch import PinnOnePeak
from utils.ISAG import SAIS
from utils.density import Uniform
from utils.gene_data import generate_peak1_samples
# torch.manual_seed(1)

model_save_path = '/home/gaozhiwei/python/FI_PINNs/models/pinn_one_peak'
data_save_path = '/home/gaozhiwei/python/FI_PINNs/data/pinn_one_peak'
img_save_path = '/home/gaozhiwei/python/FI_PINNs/figures/pinn_one_peak'

if not os.path.exists(img_save_path):
    os.makedirs(img_save_path)

if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

if not os.path.exists(data_save_path):
    os.makedirs(data_save_path)

Error = []
P_failure = []


def power_f(x, model, tol = 0.1):
        f = np.zeros(x.shape[0])
        f =  -abs(model.predict(x)[1].squeeze()) + tol
        return f

def generate_rar_samples(power_f, num_samples, num, x_interval, y_interval):
    samples = np.zeros((num, 2))
    samples[:,0] = np.random.uniform(x_interval[0], y_interval[0], num)
    samples[:,1] = np.random.uniform(x_interval[1], y_interval[1], num)
    nacp = power_f(samples)
    samples = samples[np.argsort(nacp.squeeze())]
    return samples[:num_samples]

### Repeat training
for j in range(1):

    ###Parameters for SAIS
    lb = np.array([-1, -1])
    ub = np.array([1, 1])
    mu = np.array([0,0])
    p_failure = []
    samples = []
    tol_r  = 0.1
    tol_p = 0.1
    N1 = 300 
    N2 = 1000
    p0 = 0.1
    sais = SAIS(N1, N2, p0, 500)

    error = {'SAIS_model':[],
            'Uniform_model':[], 
            'RAR_model':[]}
    epoches = 20000
    Iters = 10
    
    ###Generate data
    N_b = 200
    N_f = 2000
    X_f_train, X_b_train, u_b = generate_peak1_samples(N_b, N_f, lb, ub)
    pinn = PinnOnePeak(X_b_train, u_b, img_save_path)

    ###Initial training
    print('Start initial training----------------------------------------')
    pinn.train(X_f_train, epoches, 0)

    torch.save(pinn, os.path.join(model_save_path, 'initial_model'))
    uniform_pinn = torch.load(os.path.join(model_save_path, 'initial_model'))
    rar_pinn = torch.load(os.path.join(model_save_path, 'initial_model'))

    uniform_X_f_train = copy.deepcopy(X_f_train)
    rar_X_f_train = copy.deepcopy(X_f_train)

    error['RAR_model'].append(rar_pinn.calculate_error())
    error['Uniform_model'].append(uniform_pinn.calculate_error())
    error['SAIS_model'].append(pinn.calculate_error())

    for i in range(Iters):
        pinn.plot_error(prefix = "model:add_points" + str(i))
        uniform_pinn.plot_error(prefix = "uniform_model:add_points" + str(i))
        rar_pinn.plot_error(prefix = "rar_model:add_points" + str(i))

        power_function = partial(power_f, model = pinn, tol = tol_r)

        ###Generate new samples
        samples, p = sais.sample_uniform(power_function, Uniform, lb, ub)
        uniform_samples = np.random.uniform(-1,1, (len(samples), 2))
        rar_samples = generate_rar_samples(power_function, len(samples), 1000, lb, ub)

        X_f_train = np.vstack([X_f_train, samples])
        uniform_X_f_train = np.vstack([uniform_X_f_train, uniform_samples])
        rar_X_f_train = np.vstack([rar_X_f_train, rar_samples])

        index  = np.arange(0, len(X_f_train))
        np.random.shuffle(index)
        X_f_train = X_f_train[index, :]
        uniform_X_f_train = uniform_X_f_train[index,:]
        rar_X_f_train = rar_X_f_train[index,:]


        p_failure.append(p)
        print('--------------------------------------------------------')
        print('failure probability %.4f'%(p))
        print('current iteration: %d'%(i))
        print('--------------------------------------------------------')
        if p < tol_p:
            break
        
        with open(data_save_path + '/samples' + str(i), 'wb') as f:
            pickle.dump(samples, f)
        with open(data_save_path + '/uniform_samples' + str(i), 'wb') as f:
            pickle.dump(uniform_samples, f)
        with open(data_save_path + '/rar_samples' + str(i), 'wb') as f:
            pickle.dump(rar_samples, f)

        pinn.plot_error(add_points = samples, prefix = "model:add_points" + str(i))
        uniform_pinn.plot_error(add_points = uniform_samples, prefix = "uniform_model:add_points" + str(i))
        rar_pinn.plot_error(add_points = rar_samples, prefix = "rar_model:add_points" + str(i))

        print('Start sais retraining----------------------------------------')
        pinn.train(X_f_train, epoches, i + 1)
        pinn.plot_error(prefix="model:after training" + str(i))
        print('Start uniform retraining----------------------------------------')
        uniform_pinn.train(uniform_X_f_train, epoches, i + 1)
        uniform_pinn.plot_error(prefix="uniform_model:after training" + str(i))
        print('Start rar retraining----------------------------------------')
        rar_pinn.train(rar_X_f_train, epoches, i+1)
        rar_pinn.plot_error(prefix="rar_model:after training" + str(i))

        torch.save(pinn, os.path.join(model_save_path, 'sais_model' + str(i)))
        torch.save(uniform_pinn, os.path.join(model_save_path, 'uniform_model'+ str(i)))
        torch.save(rar_pinn, os.path.join(model_save_path, 'rar_model'+ str(i)))
        
        error['RAR_model'].append(rar_pinn.calculate_error())
        error['Uniform_model'].append(uniform_pinn.calculate_error())
        error['SAIS_model'].append(pinn.calculate_error())

    power_function = partial(power_f, model = pinn, tol = tol_r)
    samples, p = sais.sample_uniform(power_function, Uniform, lb, ub)
    p_failure.append(p)

    Error.append(error)
    P_failure.append(p_failure)
    
    
with open(os.path.join(data_save_path, 'one_peak_error'), 'wb') as f:
    pickle.dump(Error, f)
with open(os.path.join(data_save_path, 'one_peak_failure'), 'wb') as f:
    pickle.dump(P_failure, f)




    
