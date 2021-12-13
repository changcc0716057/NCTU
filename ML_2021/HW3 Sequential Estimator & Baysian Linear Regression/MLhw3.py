import argparse
import numpy as np
import matplotlib.pyplot as plt

def visualization(dataX, dataY, nbase, posterior_mean, posterior_var, noiseVar):
    x = np.linspace(start=-2, stop=2, num=400)

    # Construct the Design Matrix and transpose of Design Matrixs 
    DesignMatrix = np.ones(shape=(nbase, 400), dtype=np.float)
    for row in range(1, nbase):
        DesignMatrix[row, :] = DesignMatrix[row-1, :] * np.array(x)
    DesignMatrix_T = np.transpose(DesignMatrix)

    predict_var = np.zeros(shape=(4, 400))
    for i in range(4):
        for j in range(400):
            predict_var[i][j] = 1 / noiseVar + np.matmul(np.matmul(DesignMatrix_T[j], posterior_var[i]), np.transpose(DesignMatrix_T[j]))

    meanfunc = [np.poly1d(np.flip(posterior_mean[i])) for i in range(4)]
 
    for i in range(4):
        plt.subplot(2,2,i+1)
        ymean = meanfunc[i](x)
        yup = ymean + predict_var[i]
        ylow = ymean - predict_var[i]

        plt.plot(x, ymean, color='black')
        plt.plot(x, yup, color='red')
        plt.plot(x, ylow, color='red')
        
        if (i==1):
            plt.plot(dataX, dataY, "o", color='blue', markersize=1)
            plt.title('Predict result') 
        elif (i==2):
            plt.plot(dataX[:10], dataY[:10], "o", color='blue', markersize=1)
            plt.title('After 10 incomes') 
        elif (i==3):
            plt.plot(dataX[:50], dataY[:50], "o", color='blue', markersize=1)
            plt.title('After 50 incomes') 
        else:
            plt.title('Ground truth') 
        plt.xlim(-2,2)
        plt.ylim(-20,25)  
    plt.tight_layout()
    plt.show()

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

# parameter:
# nbase: The basis number of polynomial
# noiseVar: The variance of noise
# weights: The weight of the polynomial function
def Poly_linear_generator(nbase=1, noiseVar=1.0, weights=None):
    # Preprocessing for Data
    # Weights: 1. Get rid of redundant char; 2. Spilt the string by ','; 3. Transform to list of int; 4. Turn it to be numpy array
    # Data is uniformly distributed in range of (-1, 1)
    # Noise is from Univariate Gaussian Generator
    dataX = np.random.uniform(low=-1.0, high=1.0)
    designMatrix = np.ones(shape=(nbase, 1), dtype=np.float)
    for row in range(1, nbase):
        designMatrix[row, :] = designMatrix[row-1, :] * dataX
    Noise = Uni_Gaussian_generator(0, noiseVar)

    dataY = np.matmul(np.transpose(weights), designMatrix)[0][0] - Noise
    return dataX, dataY

# Update Current Mean and Variance by Online algorithm
# parameter: The mean and variance of univariate gaussian distribuion
def Sequential_Estimator(mean=0.0, var=1.0, eps=10**(-4)):
    print("Data point source function: N(", mean, ", ", var, ")\n",sep="")
    CurMean, CurVar = 0.0, 0.0
    dataNum = 0
    
    while True:
        # Get data point by Univariate Gaussian Data Generator given with specified mean and variance
        NewPoint = Uni_Gaussian_generator(Mean=mean, Var=var)
        dataNum += 1

        # Update the current mean and variance by Welford's online algorithm
        OldMean, OldVar = CurMean, CurVar
        CurMean = OldMean + (NewPoint - OldMean) / dataNum
        CurVar = OldVar + ((NewPoint - OldMean) * (NewPoint - CurMean) - OldVar) / dataNum

        # Show the result
        print("Add data point:", NewPoint)
        print("Mean =", CurMean, "  Variance =", CurVar)
        
        # Break if the estimates converge
        if (abs(CurMean - OldMean) <= eps and abs(CurVar - OldVar) <= eps):
            break
    
def Bayesian_Linear_Regression(precision=1.0, nbase=1, noiseVar= 1.0, weights="[1]", eps=10**-4):
    dataXs, dataYs = [], []
    ndata = 0
    recordmeans = np.zeros(shape=(4, nbase), dtype=np.float)
    recordvars = np.zeros(shape=(4, nbase, nbase), dtype=np.float)
    weights = np.array(list(map(float, weights.replace('[', '').replace(']', '').split(',')))).reshape(nbase, 1)
    recordmeans[0] = np.transpose(weights)[0]
    inv_noiseVar = 1 / noiseVar

    # Initialize the mean and variance of Prior ( variance = b^(-1)I )
    prior_mean = np.zeros(shape=(nbase, 1), dtype=np.float)
    prior_var = np.identity(nbase, dtype=np.float) / precision
    
    while True:
        # Get a new point from Polynomail basis linear model data generator     
        NewX, NewY = Poly_linear_generator(nbase, noiseVar, weights)
        dataXs.append(NewX)
        dataYs.append(NewY)
        ndata += 1

        # Construct the Design Matrix and transpose of Design Matrixs for Posterior
        DesignMatrix = np.ones(shape=(nbase, ndata), dtype=np.float)
        for row in range(1, nbase):
            DesignMatrix[row, :] = DesignMatrix[row-1, :] * np.array(dataXs)
        DesignMatrix_T = np.transpose(DesignMatrix)

        # Construct the Design Matrix and transpose of Design Matrixs for Predictive distribution
        pDesignMatrix = np.ones(shape=(nbase, 1), dtype=np.float)
        for row in range(1, nbase):
            pDesignMatrix[row, :] = pDesignMatrix[row-1, :] * NewX
        pDesignMatrix_T = np.transpose(pDesignMatrix)        

        # Find the posterior
        posterior_var = np.linalg.inv(inv_noiseVar * np.matmul(DesignMatrix, DesignMatrix_T) + precision * np.identity(nbase, dtype=np.float))
        posterior_mean = inv_noiseVar * np.matmul(np.matmul(posterior_var, DesignMatrix), np.array(dataYs).reshape(ndata, 1))

        # Calculate the mean and variance of predictive distribution
        prior_mean_T = np.transpose(prior_mean)
        predict_mean = np.matmul(prior_mean_T, pDesignMatrix)
        predict_var = 1 / inv_noiseVar + np.matmul(np.matmul(pDesignMatrix_T, prior_var), pDesignMatrix)

        print("Add data point (", NewX, ", ", NewY, "):", sep="")
        print("Posterior mean:")
        for i in range(nbase):
            print(round(posterior_mean[i][0], 10))
        print("\n\nPosterior variance:")
        for i in range(nbase):
            for j in range(nbase):
                print(round(posterior_var[i][j], 10), "   ", sep="", end="")
            print("")
        print("\n\nPredictive distribution ~ N(", predict_mean[0][0], ", ", predict_var[0][0], ")\n", sep="")

        if (ndata == 10):
            recordmeans[2] = np.transpose(posterior_mean)[0]
            recordvars[2] = posterior_var
        elif (ndata == 50):
            recordmeans[3] = np.transpose(posterior_mean)[0]
            recordvars[3] = posterior_var

        # posterior converges
        difference = posterior_mean - prior_mean
        distance = np.sqrt(np.sum(difference * difference))
        if (distance <= eps):
            recordmeans[1] = np.transpose(posterior_mean)[0]
            recordvars[1] = posterior_var
            visualization(dataXs, dataYs, nbase, recordmeans, recordvars, inv_noiseVar)
            break

        # Update the prior
        prior_mean = posterior_mean
        prior_var = posterior_var

def main(Args):
    Toggle = Args.Toggle

    # Sequential Estimator
    if Toggle == 0:
        mean = float(input("Please enter Expectation value or mean of Univariate Gaussian Generator: "))
        variance = float(input("Please enter variance of Univariate Gaussian Generator: "))
        Sequential_Estimator(mean, variance)

    # Bayesian Linear regression
    elif Toggle == 1:
        precision = float(input("Please enter the precision for initial prior: "))
        nbase = int(input("Please enter the basis number: "))
        noiseVar = float(input("Please enter variance of noise: "))
        weight = input("Please enter the weight of polynomial basis linear mode: ")
        print("")
        Bayesian_Linear_Regression(precision, nbase, noiseVar, weight)
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('Toggle', type = int, help = '0 for Sequential Estimator; 1 for Bayesian Linear regression\n')
    args = parser.parse_args()
    main(args)