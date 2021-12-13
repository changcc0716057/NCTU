import argparse
import numpy as np
import copy
import math
import matplotlib.pyplot as plt

def ReadData(path):
    Xtrain = []
    Ytrain = []
    with open(path, 'r') as f:    
        line = f.readline()
        while line:
            x, y = line.strip().split(',')
            Xtrain.append(float(x))
            Ytrain.append(float(y))
            line = f.readline()

    return np.array(Xtrain, dtype=np.float).reshape(1,len(Xtrain)), np.array(Ytrain, dtype=np.float).reshape(1,len(Ytrain))

# Matrix operations
class Matrix:
    def __init__(self):
        return
    
    # The addtion for two matrix
    def Addition(self, mA, mB):
        m, n = mA.shape[0], mA.shape[1]
        ret = np.array([[mA[i][j]+mB[i][j] for j in range(n)] for i in range(m)], dtype=np.float)
        return ret

    # The subtraction for two matrix
    def Subtraction(self, mA, mB):
        m, n = mA.shape[0], mA.shape[1]
        ret = np.array([[mA[i][j]-mB[i][j] for j in range(n)] for i in range(m)], dtype=np.float)
        return ret        

    # The multiplication for two matrix
    def Multiplication(self, mA, mB):
        m, n, p = mA.shape[0], mA.shape[1], mB.shape[1]
        ret = np.zeros(shape=(m,p), dtype=np.float)

        for i in range(m):
            for j in range(p):
                for k in range(n):
                    ret[i][j] = ret[i][j] + mA[i][k] * mB[k][j] 
        return ret

    # Create an idenetity Matrix of size n. 
    def Identity(self, size):
        I = np.array([[int(i==j) for j in range(size)] for i in range(size)], dtype=np.float)
        return I

    # The transpose of matrix
    def Transpose(self, mA):
        m, n = mA.shape[0], mA.shape[1]
        A_T = np.array([[mA[j][i] for j in range(m)] for i in range(n)], dtype=np.float)
        return A_T

    # Get the inverse of a matrix by "Gauss-Jordan elimination" 
    def Inverse(self, mA):

        n = mA.shape[0]
        # 用 Inv 來儲存 Inverse 的結果
        Inv = self.Identity(size=n)

        
        for col in range(n):
            # 透過取絕對值最大的 pivot row，以減少計算誤差，並將第 col 行與 pivot row 交換，使 pivot row 始終為於 daigonal
            pivotrow = np.argmax(abs(mA[col:,col])) + col    # 加 col 是因為我們只有取 "col:" 來找最大值，因此會少 col 行
            Inv[[pivotrow, col], :] = Inv[[col, pivotrow], :]
            mA[[pivotrow, col], :] = mA[[col, pivotrow], :]
            pivotrow = col

            # 讓 pivot row 的 diagonal 為 1
            Inv[pivotrow, :] /= mA[pivotrow][col]
            mA[pivotrow,:] /= mA[pivotrow][col]

            for i in range(n):
                # 當該 row 不為 pivot row 時，才需要做 elimination
                if (i != pivotrow):
                    Inv[i, :] -= Inv[pivotrow, :] * mA[i][col]
                    mA[i, :] -= mA[pivotrow, :] * mA[i][col]
        
        return Inv

# Create the Design Matrix of n polynomial bases for data
def CreateDesignMatrix(data, nbases):
    m = data.shape[1]
    ret = np.ones(shape=(m,nbases), dtype=np.float)

    for col in range(nbases-2, -1, -1):
        ret[:, col] = ret[:, col+1] * data

    return ret

# Regularized Least Squared Error
def LSE(Xtrain, Ytrain, nbase, lambda_):
    matrixOP = Matrix()
    A = CreateDesignMatrix(data=Xtrain, nbases=nbase)
    A_T = matrixOP.Transpose(A)
    I = matrixOP.Identity(size=nbase)
    ATA = matrixOP.Multiplication(A_T, A)
    tmp = matrixOP.Addition(ATA, lambda_*I)
    Inv = matrixOP.Inverse(tmp)
    X_LSE = matrixOP.Multiplication(matrixOP.Multiplication(Inv, A_T), matrixOP.Transpose(Ytrain))
    return X_LSE

# calculate the norm of data
def norm2(data):
    data_square = data * data
    ret = math.sqrt(sum(data_square))
    return ret

# Newton's method for optimization
def Newton(Xtrain, Ytrain, nbase, eps=10**(-7)):
    matrixOP = Matrix()
    A = CreateDesignMatrix(data=Xtrain, nbases=nbase)
    A_T = matrixOP.Transpose(A)
    ATA = matrixOP.Multiplication(A_T, A)
    ATb = matrixOP.Multiplication(A_T, matrixOP.Transpose(Ytrain))
    Hessian_Inv = matrixOP.Inverse(2 * ATA)

    trial = 0
    x0 = np.ones(shape=(nbase, 1), dtype=np.float)
    gradientX = matrixOP.Subtraction(matrixOP.Multiplication(2*ATA, x0), 2*ATb)
    xn = x0

    while trial < 1000 and norm2(gradientX) > eps:
        trial += 1
        xn = xn - matrixOP.Multiplication(Hessian_Inv, gradientX)
        gradientX = matrixOP.Subtraction(matrixOP.Multiplication(2*ATA, xn), 2*ATb)

    return xn

# Calculate Error
def Error(Xtrain, Ytrain, nbase, weight):
    matrixOP = Matrix()
    A = CreateDesignMatrix(data=Xtrain, nbases=nbase)
    Ypredict = matrixOP.Multiplication(A, weight).reshape(1, Ytrain.shape[1])
    Ydifference = Ypredict - Ytrain
    Error = np.sum(Ydifference * Ydifference)
    return Error

# Create polynomial function by weight
def weight2function(weight, nbase):
    nbase -= 1    # To match the exponentiation
    ret = (str(weight[0][0]) + "X^" + str(nbase)) if nbase > 0 else str(weight[0][0])

    for i in range(1, nbase+1):
        ret += " + " if weight[i][0] > 0 else " - "
        ret += (str(abs(weight[i][0])) + "X^" + str(nbase-i)) if (nbase-i) > 0 else str(abs(weight[i][0]))

    return ret

# The visualization of data points and fitting curve
def visualization(Xtrain, Ytrain, XLSE, XNewton):
    funcLSE = np.poly1d(XLSE.flatten())
    funcNewton = np.poly1d(XNewton.flatten())

    x = np.linspace(start=-6, stop=6, num=1200)
    yLSE = funcLSE(x)
    yNewton = funcNewton(x)
    plt.subplot(2,1,1)
    plt.plot(Xtrain, Ytrain, 'o', color='red', markersize=3)
    plt.plot(x, yLSE)
    plt.title('LSE')
    plt.subplot(2,1,2)
    plt.plot(Xtrain, Ytrain, 'o', color='red', markersize=3)
    plt.plot(x, yNewton)   
    plt.title('Newton\'s method') 
    plt.tight_layout() # 保持子圖之間有適當間距
    plt.show()
    return

def main(args):
    ### Preprocessing ###
    # read data from path, and divide them into X and Y
    path, nbase, lambda_ = args.Path, args.Nbase, args.Lambda
    Xtrain, Ytrain = ReadData(path)
    ### Preprocessing END ###
    
    ### LSE ###
    XLSE = LSE(Xtrain, Ytrain, nbase, lambda_)
    ErrorLSE = Error(Xtrain, Ytrain, nbase, XLSE)
    polyLSE = weight2function(XLSE, nbase)
    print("LSE:\nFitting line:", polyLSE)
    print("Total error:", ErrorLSE)
    ### LSE END ###

    print("")

    ### Newton's Method ###
    XNewton = Newton(Xtrain, Ytrain, nbase)
    ErrorNewton = Error(Xtrain, Ytrain, nbase, XNewton)
    polyNewton = weight2function(XNewton, nbase)
    print("Newton's Method:\nFitting line:", polyNewton)
    print("Total error:", ErrorNewton)
    ### Newton's Method END ###

    ### Visualization ###
    visualization(Xtrain, Ytrain, copy.deepcopy(XLSE), copy.deepcopy(XNewton))
    ### Visualization END ###

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('Path', type = str, help = 'Enter the path of the file\n')
    parser.add_argument('Nbase', type = int, help= 'Enter the number of polynomial bases n\n')
    parser.add_argument('Lambda', type = float, help = 'Enter the Lambda of LSE\n')
    args = parser.parse_args()
    main(args)