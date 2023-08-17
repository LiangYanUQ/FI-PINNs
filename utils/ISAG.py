import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stat
from functools import partial 
from tqdm import tqdm

class SAIS:
    """This class implements SAIS sampling method"""
    def __init__(self, N1, N2, p0, m) -> None:
        self.N1 = N1
        self.N2 = N2
        self.p0 = p0 
        self.N_b = int(N1*p0)
        self.m = m
    
    def sample_uniform(self, power_function, f, lb, ub):
        """
        This function is used to generate adaptive samples based on uniform prior distribution on bounded domains
            parameters: 
                power_function: the power function for defining the failure region
                f: the prior distribution, either uniform or normal
                lb: the left and right boundary of the domain
                ub: the upper and bottom boundary of the domain 
        """
        s = 1
        samples = np.zeros((self.N1, len(ub)))
        for i in range(len(ub)):
            samples[:, i] = np.random.uniform(lb[i], ub[i], self.N1)
        while s <= self.m:
            samples, index = self.NCPF(samples, power_function)
            if index < self.N_b:
                mu = np.mean(samples[:self.N_b], axis = 0)
                temp = samples[:self.N_b]
                sig = np.dot((temp - mu).T, (temp - mu))/(self.N_b-1)
                sample_function = partial(self.truncted_normal, mu, sig, lb, ub)
                samples = sample_function(self.N1)
            else:
                pdf = f(lb, ub).pdf(samples[:index])
                temp = np.sum(samples[:index] * pdf.reshape(-1, 1), axis = 0)
                mu = temp/np.sum(pdf)
                temp1 = samples[:index]
                sig = np.dot((temp1 - mu).T, (temp1 - mu))/(index - 1)
                sample_function = partial(self.truncted_normal, mu, sig, lb, ub)
                break
            s += 1
        print('Total evaluation number: %d'%(s))
        samples = sample_function(self.N2)
        if len(samples) == 0:
            failure_p = 0
            return [], failure_p
        samples, index = self.NCPF(samples, power_function)
        samples = samples[:index]
        p = stat.multivariate_normal(mu, sig).pdf(samples)
        failure_p = np.sum(f(lb, ub).pdf(samples).squeeze()/p)/self.N2
        return samples, failure_p
    
    def half_sample_normal(self, power_function, mu, sig, f, ub):
        """
        This function is used to generate adaptive samples based on gaussian prior distribution on unbounded domains
        parameters: 
                power_function: the power function for defining the failure region
                f: the prior distribution, either uniform or normal
                mu: the mean vector for the prior distribution
                sig: the covariance for the prior distribution
                ub: the upper and bottom boundary of the domain 

        """
        s = 1
        f = f(mu, sig)
        samples = self.half_tructed_normal(mu, sig, ub, self.N1)
        while s <= self.m:
            samples, index = self.NCPF(samples, power_function)
            if index < self.N_b:
                mu = np.mean(samples[:self.N_b], axis = 0)
                temp = samples[:self.N_b]
                sig = np.dot((temp - np.mean(temp, axis = 0)).T, (temp - np.mean(temp, axis = 0)))
                sample_function = partial(self.half_tructed_normal, mu, sig, ub)
                samples = sample_function(self.N1)
    
            else:
                pdf = f.pdf(samples[:index])
                temp = np.sum(samples[:index] * pdf.reshape(-1, 1), axis = 0)
                mu = temp/np.sum(pdf)
                temp1 = samples[:index]
                sig = np.dot((temp1 - np.mean(temp1, axis = 0)).T, (temp1 - np.mean(temp1, axis = 0)))/(index - 1)
                sample_function = partial(self.half_tructed_normal, mu, sig, ub)
            s += 1
        print('Iteration: %d'%(s))
        samples = sample_function(self.N2)
        if len(samples) == 0:
            failure_p = 0
            return samples, failure_p
        samples, index = self.NCPF(samples, power_function)
        samples = samples[:index]
        p = stat.multivariate_normal(mu, sig).pdf(samples)
        failure_p = np.sum(f.pdf(samples).squeeze()/p)/self.N2
        return samples, failure_p
    
    def whole_sample_normal(self, power_function, mu, sig, f):
        """
        This function is used to generate adaptive samples based on gaussian prior distribution on unbounded domains
        parameters: 
                power_function: the power function for defining the failure region
                f: the prior distribution, either uniform or normal
                mu: the mean vector for the prior distribution
                sig: the covariance for the prior distribution 
        """
        s = 1
        f = f(mu, sig)
        samples = f.sample(self.N1)
        while s <= self.m:
            samples, index = self.NCPF(samples, power_function)
            if index < self.N_b:
                mu = np.mean(samples[:self.N_b], axis = 0)
                temp = samples[:self.N_b]
                sig =  np.dot((temp - np.mean(temp, axis = 0)).T, (temp - np.mean(temp, axis = 0)))/(self.N_b - 1)
                # sig = sig/sig[1,1]
                sample_function = partial(np.random.multivariate_normal, mu, sig)
                samples = sample_function(self.N1)
    
            else:
                pdf = f.pdf(samples[:index])
                temp = np.sum(samples[:index] * pdf.reshape(-1, 1), axis = 0)
                mu = temp/np.sum(pdf)
                temp1 = samples[:index]
                sig = s*np.dot((temp1 - np.mean(temp1, axis = 0)).T, (temp1 - np.mean(temp1, axis = 0)))/(index - 1)
                # sig = sig/np.min(sig)
                sample_function = partial(np.random.multivariate_normal, mu, sig)
                break
            s += 1
        print('Iteration: %d'%(s))
        samples = sample_function(self.N2)
        if len(samples) == 0:
            failure_p = 0
            return samples, failure_p 
        samples, index = self.NCPF(samples, power_function)
        samples = samples[:index]
        p = stat.multivariate_normal(mu, sig).pdf(samples)
        failure_p = np.sum(f.pdf(samples).squeeze()/p)/self.N2
        return samples, failure_p
    
    def NCPF(self, samples, power_f):
        lam = np.max(abs(power_f(samples)))
        nacp = power_f(samples)/lam
        samples = samples[np.argsort(nacp.squeeze())]
        index = np.sum(nacp<0)
        return samples, index
    
    def truncted_normal(self, mu, sig, lb, ub, num_samples):
        num = num_samples * 2
        samples = np.random.multivariate_normal(mu, sig, num)
        index = True
        for i in range(len(lb)):
            index = index & (lb[i] < samples[:,i]) & (ub[i] > samples[:,i])
        i = 3
        while sum(index) < num_samples:
            num = num_samples * i
            samples = np.random.multivariate_normal(mu, sig, num)
            index = 1
            for i in range(len(lb)):
                index = index & (lb[i] < samples[:,i]) & (ub[i] > samples[:,i])
            i = i+1
        choice = np.random.choice(np.arange(0, sum(index)), num_samples, replace=False)
        return samples[index][choice]
    
    def half_tructed_normal(self, mu, sig, ub, num_samples):
        num = num_samples * 2
        samples = np.random.multivariate_normal(mu, sig, num)
        index = (samples[:,1] < ub[1]) & (samples[:, 1] > ub[0])
        i = 3
        while sum(index) < num_samples:
            num = num_samples * i
            samples = np.random.multivariate_normal(mu, sig, num)
            index = (samples[:,1] < ub[1]) & (samples[:, 1] > ub[0])
            i = i+1
        choice = np.random.choice(np.arange(0, sum(index)), num_samples, replace=False)
        return samples[index][choice]
