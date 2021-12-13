import argparse
import numpy as np
import matplotlib.pyplot as plt
import math

np.set_printoptions(threshold=np.inf)
# Marsaglia polar method
# parameter: The mean and variance of univariate gaussian distribuion
def Uni_Gaussian_generator(Mean=0.0, Var=1.0):
    while True:
        # U and V are drawn from the unifrom (-1,1) distribution
        numU = np.random.uniform(low=-1.0, high=1.0)
        numV = np.random.uniform(low=-1.0, high=1.0)
        # S = U^2 + V^2
        squareSUM = numU ** 2 + numV ** 2

        # If S is greater or equal to 1, then the method starts over
        if squareSUM < 1:
            std_gaussian_value = numU * np.sqrt(-2 * np.log(squareSUM) / squareSUM)
            Uni_Gaussian_value = Mean + np.sqrt(Var) * std_gaussian_value
            return Uni_Gaussian_value

def sigmoid(x):
    sx = (x >= 0) * (1.0 / (1 + np.exp(-x))) + (x < 0) * (np.exp(x) / (1 + np.exp(x)))
    return sx

def Hessian(DesignMat, weight):
    phi_W = np.matmul(DesignMat, weight)
    z = sigmoid(phi_W).flatten()
    diagonal = z * (1 - z)
    Dmat = np.diag(diagonal)

    # Hessian = phi_T * D * phi
    return np.matmul(np.matmul(np.transpose(DesignMat), Dmat), DesignMat)

def Gradient(DesignMat, ydata, weight):
    # 1 / (1 + exp(-W_T * X_i)) - y
    phi_W = np.matmul(DesignMat, weight)
    z = sigmoid(phi_W) - ydata
        
    # Gradient = phi_T * (sigmoid - y)
    return np.matmul(np.transpose(DesignMat), z)

def createDesignMat(xfeature, yfeature):
    ndata = xfeature.shape[0]
    desingMat = np.ones(shape=(ndata, 3), dtype=np.float)
    desingMat[:, 0], desingMat[:, 1] = xfeature, yfeature
    return desingMat

# calculate the norm of data
def norm2(data):
    data_square = data * data
    ret = math.sqrt(sum(data_square))
    return ret

# Newton's method
def Newton(DesignMat, Lable, learning_rate=10**(-2), eps=10**(-6)):
    ### Initialize ###
    # x0 is initial weight
    trial = 0
    xn = np.zeros(shape=(3, 1), dtype=np.float)
    Hes = Hessian(DesignMat, xn)
    grad = Gradient(DesignMat, Lable, xn)
    try:
        Hes_Inv = np.linalg.inv(Hes)
        update = np.matmul(Hes_Inv, grad)
    except np.linalg.LinAlgError:
        update = learning_rate * grad

    while trial < 100000 and norm2(update) > eps:
        trial += 1
        xn = xn - update
        Hes = Hessian(DesignMat, xn)
        grad = Gradient(DesignMat, Lable, xn)
        try:
            Hes_Inv = np.linalg.inv(Hes)
            update = np.matmul(Hes_Inv, grad)
        except np.linalg.LinAlgError:
            update = learning_rate * grad

    return xn

def Gradient_Descent(DesignMat, Lable, learning_rate=10**(-1), eps=10**(-6)):
    ### Initialize ###
    # x0 is initial weight
    trial = 0
    xn = np.zeros(shape=(3, 1), dtype=np.float)
    grad = Gradient(DesignMat, Lable, xn)

    while trial < 100000 and norm2(grad) > eps:
        trial += 1
        xn = xn - learning_rate * grad
        grad = Gradient(DesignMat, Lable, xn)

    return xn    

def predict(DesignMat, Xfeature, Yfeature, Label, weight):
    predictValue = np.matmul(DesignMat, weight).flatten()
    p0_X, p0_Y, label0 = Xfeature[predictValue < 0], Yfeature[predictValue < 0], Label[predictValue < 0]
    p1_X, p1_Y, label1 = Xfeature[predictValue >= 0], Yfeature[predictValue >= 0], Label[predictValue >= 0]
    return p0_X, p0_Y, p1_X, p1_Y, label0, label1       

def PrintInfo(types, weight, label0, label1):
    print(types, ":\n\n", "w:\n", sep="", end="")
    print(weight[2][0], '\n', weight[0][0], '\n', weight[1][0], '\n', sep="")
    print("Confusion Matrix:\n             Predict cluster 1 Predict cluster 2")
    print("Is cluster 1        ", np.sum(label0 == 0), "               ", np.sum(label1 == 0), sep="")
    print("Is cluster 2        ", np.sum(label0 == 1), "               ", np.sum(label1 == 1), sep="")
    print("")
    print("Sensitivity (Successfully predict cluster 1):", np.sum(label0 == 0) / 50)
    print("Specificity (Successfully predict cluster 2):", np.sum(label1 == 1) / 50)

def main(Args):
    # Generating Datas for D1 and D2 by using Univariate Gaussian Data Generator
    d0_X, d0_Y, d1_X, d1_Y = [], [], [], []
    for _ in range(Args.NumOfData):
        d0_X.append(Uni_Gaussian_generator(Mean=Args.mx1, Var=Args.vx1))
        d0_Y.append(Uni_Gaussian_generator(Mean=Args.my1, Var=Args.vy1))
        d1_X.append(Uni_Gaussian_generator(Mean=Args.mx2, Var=Args.vx2))
        d1_Y.append(Uni_Gaussian_generator(Mean=Args.my2, Var=Args.vy2))

    ### Preprocessing ###
    Xfeature = np.append(d0_X, d1_X)
    Yfeature = np.append(d0_Y, d1_Y)
    Label = np.append(np.zeros(shape=(Args.NumOfData), dtype=int), np.ones(shape=(Args.NumOfData), dtype=int)).reshape(2 * Args.NumOfData, 1)
    DesignMat = createDesignMat(Xfeature, Yfeature)
    ### Preprocessing End ###

    ### Training ###
    NewtonWeight = Newton(DesignMat, Label)
    GradientWeight = Gradient_Descent(DesignMat, Label)
    ### Trainint End ###
    
    ### Prediction ###
    n0_X, n0_Y, n1_X, n1_Y, nlabel0, nlable1 = predict(DesignMat, Xfeature, Yfeature, Label, NewtonWeight)    
    g0_X, g0_Y, g1_X, g1_Y, glabel0, glabel1 = predict(DesignMat, Xfeature, Yfeature, Label, GradientWeight)
    ### Prediction End ###

    ### Print Info ###
    PrintInfo("Gradient descent", GradientWeight, glabel0, glabel1)
    print("\n------------------------------------------------------------------")
    PrintInfo("Newton\'s metho", NewtonWeight, nlabel0, nlable1)
    ### Print Info End ###

    ### Visualization ###
    plt.subplot(1,3,1)
    plt.plot(d0_X, d0_Y, 'o', color='red', markersize=2)
    plt.plot(d1_X, d1_Y, 'o', color='blue', markersize=2)
    plt.title('Ground truth')

    plt.subplot(1,3,2)
    plt.plot(g0_X, g0_Y, 'o', color='red', markersize=2)
    plt.plot(g1_X, g1_Y, 'o', color='blue', markersize=2)
    plt.title('Gradient descent')

    plt.subplot(1,3,3)
    plt.plot(n0_X, n0_Y, 'o', color='red', markersize=2)
    plt.plot(n1_X, n1_Y, 'o', color='blue', markersize=2)
    plt.title('Newton\'s method')
    plt.show()
    ### Visualization End ###

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('NumOfData', type = int, help = 'Please enter the number of data points\n')
    parser.add_argument('mx1', type = float, help = 'Please enter the mean of x1\n')
    parser.add_argument('vx1', type = float, help = 'Please enter the variance of x1\n')
    parser.add_argument('my1', type = float, help = 'Please enter the mean of y1\n')
    parser.add_argument('vy1', type = float, help = 'Please enter the variance of y1\n')
    parser.add_argument('mx2', type = float, help = 'Please enter the mean of x2\n')
    parser.add_argument('vx2', type = float, help = 'Please enter the variance of x2\n')
    parser.add_argument('my2', type = float, help = 'Please enter the mean of y2\n')
    parser.add_argument('vy2', type = float, help = 'Please enter the variance of y2\n')
    args = parser.parse_args()
    main(args)