import argparse
import csv
from math import cos
import numpy as np
import time
from libsvm.svmutil import *

Xtrain_path, Ytrain_path = "X_train.csv", "Y_train.csv"
Xtest_path, Ytest_path = "X_test.csv", "Y_test.csv"
task_msg = {1: "Part 1: Compare the performance between different kernel functions.",
        2: "Part 2: Use C-SVC and perform grid search to find best hyperparameter.",
        3: "Part 3: Use linear kernel + RBF kernel, and compare the performance."}
kernel_type = {0: "linear",
            1: "polynomial",
            2: "radial basis function (RBF)",
            4: "precomputed kernel"}
kernel_function = {0: "u'*v",
            1: "(gamma*u'*v + coef0)^degree",
            2: "exp(-gamma*|u-v|^2)",
            4: "u'*v + exp(-gamma*|u-v|^2)"}
            
def ReadData(path):
    """
    Read Datas by path.
    Args:
        path: The path of the data file.

    Return Value:
        ndarray of datas.
    """
    with open(path, newline='') as csvfile:
        data = np.array(list(csv.reader(csvfile)))
    return data

def linear_kernel(X_M, X_N):
    """
    kernel with linear term.
    Args:
        X_M: Array of M values.
        X_N: Array of N values.

    Return Value:
        (M x N) matrix.
    """
    return np.matmul(X_M, np.transpose(X_N))

def RBF_kernel(X_M, X_N, gamma=1.0):
    """
    RBF kernel.
    Args:
        X_M: Array of M values.
        X_N: Array of N values.
        gamma: hyper-parameter for quadratic term.  

    Return Value:
        (M x N) matrix.
    """
    return np.exp((-1) * gamma * (np.sum(X_M ** 2, axis=1).reshape(-1, 1) + \
        np.sum(X_N ** 2, axis=1) - 2 * np.matmul(X_M, np.transpose(X_N))))

def train(Xtrain, Ytrain, option='', iskernel=False):
    """
    Training a SVM model.
    Args:
        Xtrain: Features of training datas.
        Ytrain: Labels of training datas.
        iskernel: True if the kernel is user-defined.
        option: hyperparameters for training.

    Return Value:
        A SVM model if not use cross validation.
        Accuracy if use cross validation.
    """
    prob = svm_problem(Ytrain, Xtrain, iskernel)
    param = svm_parameter(option)
    res = svm_train(prob, param)
    return res

def cross_validation_train(Xtrain, Ytrain, option='', best_acc=0, best_option='', iskernel=False):
    """
    Training a SVM model with validation.
    Args:
        Xtrain: Features of training datas.
        Ytrain: Labels of training datas.
        iskernel: True if the kernel is user-defined.
        option: Hyperparameters for training.

    Return Value:
        best_acc: The current best accuracy.
        best_option: The option for best accuracy. 
    """
    new_acc = train(Xtrain, Ytrain, option, iskernel)
    if (new_acc > best_acc):
        best_acc = new_acc
        best_option = option
    return best_acc, best_option

def predict(Xtest, Ytest, model):
    """
    Doing the prediction for testing data.
    Args:
        Xtest: Features of testing datas.
        Ytest: Labels of testing datas.
        model: The model which is used to predict.

    Return Value:
        p_acc: a tuple including accuracy (for classification), mean-squared 
        error, and squared correlation coefficient (for regression).
    """  
    return svm_predict(Ytest, Xtest, model)[1]

def GridSearch(Xtrain, Ytrain, ktype):
    """
    Use grid search to find the best parameter for training.
    Args:
        Xtrain: Features of training datas.
        Ytrain: Labels of training datas.
        ktype: The type of the kernel.

    Return Value:
        opt_acc: The optimal accuracy in cross validation set.
        opt_option: The option for optimal accuracy.         
    """     
    k_fold = 5
    opt_acc = 0
    opt_option = ''
    gamma = [ 10 ** power for power in range(-3, 2) ]
    coef0 = [ 10 ** power for power in range(-1, 3) ]
    cost = [ 10 ** power for power in range(-2, 3) ]
    degree = [ deg for deg in range(2, 5) ]

    ### grid search start ###
    print("Kernel Type =", kernel_type[ktype])
    print("Kernel Function =", kernel_function[ktype])
    print("SVM type = C-SVC")

    if ktype == 0:
        for c in cost:
            print('Current Setting: Cost = {}'.format(c))
            cur_option = '-s 0 -t {} -c {} -v {} -q'.format(str(ktype), str(c), str(k_fold))
            opt_acc, opt_option = cross_validation_train(Xtrain, Ytrain, cur_option, opt_acc, opt_option, False)
    elif ktype == 1:
        for c in cost:
            for d in degree:
                for g in gamma:
                    for coef in coef0:
                        print('Current Setting: Cost = {}, Degree = {}, Gamma = {}, Coef0 = {}'.format(c, d, g, coef))
                        cur_option = '-s 0 -t {} -c {} -d {} -g {} -r {} -v {} -q'.format(str(ktype), str(c), str(d), str(g), str(coef), str(k_fold))
                        opt_acc, opt_option = cross_validation_train(Xtrain, Ytrain, cur_option, opt_acc, opt_option, False)            
    elif ktype == 2:
        for c in cost:
            for g in gamma:
                print('Current Setting: Cost = {}, Gamma = {}'.format(c, g))
                cur_option = '-s 0 -t {} -c {} -g {} -v {} -q'.format(str(ktype), str(c), str(g), str(k_fold))
                opt_acc, opt_option = cross_validation_train(Xtrain, Ytrain, cur_option, opt_acc, opt_option, False)
    else:
        linear_k = linear_kernel(Xtrain, Xtrain)
        for g in gamma:
            RBF_k = RBF_kernel(Xtrain, Xtrain, g)
            for c in cost:
                print('Current Setting: Cost = {}, Gamma = {}'.format(c, g))
                kernel = np.hstack((np.arange(1, 5001).reshape(-1,1), linear_k + RBF_k))
                cur_option = '-s 0 -t {} -c {} -g {} -v {} -q'.format(str(ktype), str(c), str(g), str(k_fold))
                opt_acc, opt_option = cross_validation_train(kernel, Ytrain, cur_option, opt_acc, opt_option, True)
                if cur_option == opt_option:
                    opt_option = '{} {} {}'.format(opt_option, c, g)

    ### grid search end ###
    return opt_acc, opt_option

def run(Xtrain, Ytrain, Xtest, Ytest, Mode):
    ### Print basic information ###
    print("\n-------------[Log]: {}-------------".format(task_msg[Mode]))

    if Mode == 1:
        print("-------------[Log]: Basic C-SVC for three types of kernel-------------")
        for ktype in range(0,3):
            print("Kernel Type =", kernel_type[ktype])
            print("Kernel Function =", kernel_function[ktype])
            print("SVM type = C-SVC")
            
            start = time.time()
            SVMmodel = train(Xtrain, Ytrain, '-s 0 -t {} -q'.format(str(ktype)), False)
            PredictRes = predict(Xtest, Ytest, SVMmodel)
            stop = time.time()
            
            print("Mean-Squared Error (MSE) =", PredictRes[1])
            print("Time spent = ", stop-start, "s\n", sep="")

    elif Mode == 2:
        print("-------------[Log]: Grid Search with 5-fold cross validation-------------")
        for ktype in range(0,3):
            print("-------------[Log]: Grid Search for {}-------------".format(kernel_type[ktype]))
            best_acc, best_option = GridSearch(Xtrain, Ytrain, ktype)
            print("-------------[Log]: Grid Search for {} End-------------".format(kernel_type[ktype]))
            print('Best Cross Validation Accuracy = {}%'.format(best_acc))
            print('Best option = \"{}\"'.format(best_option))
            print("Start to use best option to train...")
            best_option = best_option.replace("-v 5 ", "", 1)
            SVMmodel = train(Xtrain, Ytrain, best_option, False)
            PredictRes = predict(Xtest, Ytest, SVMmodel)
            print("Mean-Squared Error (MSE) =", PredictRes[1], "\n")     

    elif Mode == 3:
        ktype = 4
        print("-------------[Log]: Grid Search with 5-fold cross validation-------------")
        print("-------------[Log]: Grid Search for {}-------------".format(kernel_type[ktype]))
        best_acc, best_option = GridSearch(Xtrain, Ytrain, ktype)
        print("-------------[Log]: Grid Search for {} End-------------".format(kernel_type[ktype]))
        print('Best Cross Validation Accuracy = {}%'.format(best_acc))
        print('Best option = \"{}\"'.format(best_option))
        print("Start to use best option to train...")
        gamma, cost = best_option.split()[-2], best_option.split()[-1]
        
        start = time.time()
        linear_k = linear_kernel(Xtrain, Xtrain)
        RBF_k = RBF_kernel(Xtrain, Xtrain, float(gamma))
        kernel = np.hstack((np.arange(1, 5001).reshape(-1,1), linear_k + RBF_k))
        SVMmodel = train(kernel, Ytrain, '-s 0 -t {} -c {} -g {} -q'.format(str(ktype), str(cost), str(gamma)), True)

        linear_test = np.transpose(linear_kernel(Xtrain, Xtest))
        RBF_test = np.transpose(RBF_kernel(Xtrain, Xtest, float(gamma)))
        kernel_test = np.hstack((np.arange(1, 2501).reshape(-1,1), linear_test + RBF_test))        
        PredictRes = predict(kernel_test, Ytest, SVMmodel)
        stop = time.time()
            
        print("Mean-Squared Error (MSE) =", PredictRes[1])
        print("Time spent = ", stop-start, "s\n", sep="")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('Mode', type = int, help = '1: basic, 2: C-SVC, 3: linear + RBF kernel\n')
    args = parser.parse_args()
    Xtrain, Xtest = ReadData(Xtrain_path).astype(np.float), ReadData(Xtest_path).astype(np.float)
    Ytrain, Ytest = ReadData(Ytrain_path).astype(np.int).flatten(), ReadData(Ytest_path).astype(np.int).flatten()
    run(Xtrain, Ytrain, Xtest, Ytest, args.Mode)