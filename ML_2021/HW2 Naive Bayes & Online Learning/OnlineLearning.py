import argparse
import math

def ReadData(path):
    traindata = []
    with open(path, 'r') as f:    
        line = f.readline().strip()
        while line:
            traindata.append(line)
            line = f.readline().strip()
    return traindata

class Beta:
    """ Initialize Beta Distribution:
            Variables:
                pfrequency: The number of occurence of (label, pixel, bin) = (i, j, k)
                pcount: The total number of occurence of (label, pixel) = (i, j)
                pprobability: The probability of occurence of (label, pixel, bin) = (i, j, k)
    """
    def __init__(self, alpha=0, beta=0, theta=0):
        self.alpha = alpha
        self.beta = beta
        self.theta = theta

    def gamma(self, coefficient):
        return math.factorial(coefficient-1)

    def update(self, win, lose):
        self.alpha += win
        self.beta += lose

    def calprior(self):
        prior = (self.theta ** (self.alpha - 1)) * ((1 - self.theta) ** (self.beta - 1)) * self.gamma(self.alpha + self.beta) / (self.gamma(self.alpha) * self.gamma(self.beta))
        return prior

def combination(total, win):
    numerator = math.factorial(total) / math.factorial(total - win)
    denominator = math.factorial(win)
    return numerator / denominator

def binomial(theta, total, win):
    C = combination(total, win)
    probability = C * (theta ** win) * ((1 - theta) ** (total - win))
    return probability

def main(Args):
    ### Preprocessing ###
    # read data from path
    path, beta_a, beta_b = Args.Path, Args.a, Args.b
    trainData = ReadData(path)
    ### Preprocessing END ###

    B = Beta(beta_a, beta_b, 0.5)
    for i in range(len(trainData)):
        # Get the information of string
        total = len(trainData[i])
        win = trainData[i].count('1')
        lose = trainData[i].count('0')
        likelihood = binomial(win/total, total, win)

        print("case ", i+1, ": ", trainData[i], sep="")
        print("Likelihood: ", likelihood, sep="")
        print("Beta prior:     a = ", B.alpha, " b = ", B.beta, sep="")
        B.update(win, lose)
        print("Beta posterior: a = ", B.alpha, " b = ", B.beta, sep="")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('Path', type = str, help = 'Enter the path of the file\n')
    parser.add_argument('a', type = int, help = 'Enter parameter a for the initial beta prior\n')
    parser.add_argument('b', type = int, help = 'Enter parameter b for the initial beta prior\n')
    args = parser.parse_args()
    main(args)