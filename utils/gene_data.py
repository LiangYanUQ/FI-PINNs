import numpy as np
from pyDOE import lhs


def generate_peak1_samples(N_b, N_f, lb, ub):
    np.random.seed(1)
    u_true = lambda x, y: np.exp(-1000*((x-0.5)**2 + (y - 0.5)**2))
    X_f = lb + (ub-lb)*lhs(2, N_f)
    x_lb = np.random.uniform(-1,1,(N_b//4, 1))
    u_lb = u_true(-1, x_lb)
    x_rb = np.random.uniform(-1,1,(N_b//4, 1))
    u_rb = u_true(1, x_rb)
    x_ub = np.random.uniform(-1,1,(N_b//4, 1))
    u_ub = u_true(x_ub, 1)
    x_bb = np.random.uniform(-1,1, (N_b//4, 1))
    u_bb = u_true(x_bb, -1)
    X_lb = np.hstack([-np.ones((N_b//4, 1)), x_lb])
    X_ub = np.hstack([x_ub, np.ones((N_b//4, 1))])
    X_rb = np.hstack([np.ones((N_b//4, 1)), x_rb])
    X_bb = np.hstack([x_bb, -np.ones((N_b//4, 1))])
    X_b_train = np.vstack([X_lb, X_ub, X_rb, X_bb])
    u_b = np.vstack([u_lb, u_ub, u_rb, u_bb])
    index = np.arange(0, N_b)
    np.random.shuffle(index)
    X_b_train = X_b_train[index]
    u_b = u_b[index]
    return X_f, X_b_train, u_b
    


