import numpy as np
import scipy.stats as stat

class Uniform:
    """
    This class generates the uniform prior distribution
    """
    def __init__(self, lb, ub) -> None:
        self.lb = lb 
        self.ub = ub 
        self.dim = len(ub)

    def pdf(self, samples):
        V = 1 
        for i in range(self.dim):
            V = V * (self.ub[i] - self.lb[i])
        return np.ones((samples.shape[0], 1))/V
    
    def sample(self, num_samples):
        samples = np.zeros((num_samples, self.dim))
        for i in range(self.dim):
            samples[:, i] = np.random.uniform(self.lb[i], self.ub[i], num_samples)
        return samples 


class Normal:
    """This class generates the normal prior distribution"""
    def __init__(self, mu, sig) -> None:
        self.mu = mu 
        self.sig = sig 

    def pdf(self, samples):
        return stat.multivariate_normal(self.mu, self.sig).pdf(samples)
    
    def sample(self, num_samples):
        return np.random.multivariate_normal(self.mu, self.sig, num_samples)




