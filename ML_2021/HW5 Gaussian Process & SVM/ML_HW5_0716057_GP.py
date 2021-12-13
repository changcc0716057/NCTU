import argparse
import numpy as np
import matplotlib.pyplot as plt 
import scipy.optimize as opt
InputPath = "input.data"

def ReadData(path):
    """
    Read Datas by path.
    Args:
        path: The path of the data file.

    Return Value:
        (1 x N) matrix of X.
        (1 x N) matrix of Y.
    """
    Xs, Ys = [], []
    with open(path, 'r') as f:    
        line = f.readline()
        while line:
            x, y = line.strip().split(' ')
            Xs.append(float(x))
            Ys.append(float(y))
            line = f.readline()
    return np.array(Xs, dtype=np.float).reshape(-1, 1), np.array(Ys, dtype=np.float).reshape(-1, 1)

def kernel(X_M, X_N, sigma=1.0, gamma=1.0):
    """
    kernel with quadratic term, linear term, and constant.
    Args:
        X_M: Array of M values.
        X_N: Array of N values.
        sigma: hyper-parameter for RBF kernel.
        gamma: hyper-parameter for quadratic term.

    Return Value:
        (M x N) matrix.
    """
    MxN = np.matmul(X_M, np.transpose(X_N))
    RBF_k = sigma * np.exp((-1) * gamma * (X_M ** 2 + np.sum(X_N ** 2, axis=1) - 2 * MxN))
    return RBF_k

def prediction(Xtrain, Xtest, Ytrain, noise_var=1.0, sigma=1.0, gamma=1.0):
    """
    Prediction by Gaussian Process Regression.
    Args:
        Xtrain: Array of training data for X values. (M x 1)
        Xtest: Array of testing data for X values.   (N x 1)
        Ytrain: Array of training data for Y values. (M x 1)
        noise_var: Variance of noise.
        sigma, gamma: hyper-parameters for kernel.

    Return Value:
        mean: Array of mean values of each data in Xtest. (N x 1)
        variance: Array of variance values of each data in Xtest. (N x N)
    """
    ## Covar: The relation between all training data points (M x M)
    ## Covar_t: The relation between all training data points and all testing data points (M x N)
    ## Covar_tt: The relation between all testing data points (N x N)
    Covar = kernel(Xtrain, Xtrain, sigma, gamma) + noise_var * np.identity(Xtrain.shape[0])
    Covar_t = kernel(Xtrain, Xtest, sigma, gamma)
    Covar_tt = kernel(Xtest, Xtest, sigma, gamma) + noise_var * np.identity(Xtest.shape[0])
    Covar_inv = np.linalg.inv(Covar)

    means = np.matmul(np.matmul(np.transpose(Covar_t), Covar_inv), Ytrain)
    variances = Covar_tt -  np.matmul(np.matmul(np.transpose(Covar_t), Covar_inv), Covar_t)

    return means, variances

def NegativeMarginalLogLikelihood(param, Xtrain, Ytrain, noise_var):
    """
    Calculate negative marginal log likelihood.
    Args:
        param: includes sigma, gamma, which are needed to optimize.
        Xtrain: Array of training data for X values. (M x 1)
        Ytrain: Array of training data for Y values. (M x 1)
        noise_var: Variance of noise.

    Return Value:
        nmll: negative marginal log likelihood value for current parameter.
    """
    sigma, gamma = param
    Covar = kernel(Xtrain, Xtrain, sigma, gamma) + noise_var * np.identity(Xtrain.shape[0])
    Covar_inv = np.linalg.inv(Covar)
    Ytrain_t = np.transpose(Ytrain)
    nmll = (1/2) * np.log(np.linalg.det(Covar)) + (Xtrain.shape[0] / 2) * np.log(2 * np.pi)
    nmll += (1/2) * (np.matmul(np.matmul(Ytrain_t, Covar_inv), Ytrain)[0][0])
    return nmll

def run(Xtrain, Ytrain, Mode):
    """
    Main function.
    Args:
        Xtrain: Array of X values.
        Ytrain: Array of Y values.
    
    Return Value:
        NULL
    """
    ### Training and Prediction ###
    noise_var = 1/5
    Xtest = np.arange(start=-60, stop=60, step=0.1).reshape(-1, 1)
    if Mode == 0:
        sigma, gamma = 1.0, 1.0
        means, variances = prediction(Xtrain, Xtest, Ytrain, noise_var, sigma, gamma)
    else:
        initial_guess = [1.0, 1.0]
        result = opt.minimize(fun=NegativeMarginalLogLikelihood, x0=initial_guess, \
            bounds = ((1e-8, 1e8), (1e-8, 1e8)), args = (Xtrain, Ytrain, noise_var))
        if result.success:
            sigma, gamma = result.x[0], result.x[1]
            means, variances = prediction(Xtrain, Xtest, Ytrain, noise_var, sigma, gamma)
        else:
            raise ValueError(result.message)
    ### End ###

    ### Visualization ###
    CI = np.sqrt(np.diag(variances)) * 1.96
    Xtest = Xtest.flatten()
    means = means.flatten()
    UpperBound, LowerBound = means + CI, means - CI
    plt.plot(Xtest, means, color='blue')
    plt.fill_between(Xtest, UpperBound, LowerBound, where= UpperBound >= LowerBound, \
    facecolor = "yellow", edgecolor = "blue")
    plt.plot(Xtrain, Ytrain, "o", color='green', markersize=5)  

    plt.title('hyperparam: \nsigma = {:.6f}, gamma = {:.6f}'.format(sigma, gamma))
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.xlim(-60, 60)
    plt.show()
    ### End ###

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('Mode', type = int, help = '0: Not optimize, 1: optimize\n')
    args = parser.parse_args()
    ### Preprocessing ###
    # read data from path, and divide them into X and Y
    Xdata, Ydata = ReadData(InputPath)
    ### Preprocessing END ###
    
    run(Xdata, Ydata, args.Mode)