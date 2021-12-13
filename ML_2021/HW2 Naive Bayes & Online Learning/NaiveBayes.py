import gzip
import argparse
import numpy as np

TrainImagePath = 'C:/Users/jackson/Desktop/chiacheng/visualstudiocode/nctu110fall/ML/hw2/train-images-idx3-ubyte.gz'
TrainLablePath = 'C:/Users/jackson/Desktop/chiacheng/visualstudiocode/nctu110fall/ML/hw2/train-labels-idx1-ubyte.gz'
TestImagePath = 'C:/Users/jackson/Desktop/chiacheng/visualstudiocode/nctu110fall/ML/hw2/t10k-images-idx3-ubyte.gz'
TestLablePath = 'C:/Users/jackson/Desktop/chiacheng/visualstudiocode/nctu110fall/ML/hw2/t10k-labels-idx1-ubyte.gz'
np.set_printoptions(threshold=np.inf)

def readImage(path):
    with gzip.open(path, 'r') as f:
        # first 16 bytes: 4 for magic number, 4 for number of Images, 4 for number of rows, and 4 for number of columns
        mNumber = int.from_bytes(bytes=f.read(4), byteorder='big', signed=False)
        nImage = int.from_bytes(bytes=f.read(4), byteorder='big', signed=False)
        nrow = int.from_bytes(bytes=f.read(4), byteorder='big', signed=False)
        ncol = int.from_bytes(bytes=f.read(4), byteorder='big', signed=False)

        # other bytes: pixels of Image (each pixel is an unsigned byte)
        Images = np.frombuffer(f.read(), dtype=np.uint8).reshape(nImage, nrow, ncol)
        Xdata = Image(mNumber, nImage, nrow, ncol, Images)
    return Xdata

def readLable(path):
    with gzip.open(path, 'r') as f:
        # first 8 bytes: 4 for magic number, 4 for number of Lables
        mNumber = int.from_bytes(bytes=f.read(4), byteorder='big', signed=False)
        nLable = int.from_bytes(bytes=f.read(4), byteorder='big', signed=False)

        # other bytes: lables (each lable is an unsigned byte)
        Lables = np.frombuffer(f.read(), dtype=np.uint8)
        Ydata = Lable(mNumber, nLable, Lables)
    return Ydata

class Image:
    def __init__(self, mNumber, nImage, nrow, ncol, Images):
        self.MagicNumber = mNumber
        self.nImage = nImage
        self.nrow = nrow
        self.ncol = ncol
        self.Images = Images

class Lable:
    def __init__(self, mNumber, nLable, Lables):
        self.MagicNumber = mNumber
        self.nLable = nLable
        self.Lables = Lables

class DiscreteNBClassifer:
    """Initialize a Naive Bayes Classifier for Discrete Mode

    Variables:
        pfrequency: The number of occurence of (label, pixel, bin) = (i, j, k)
        pprobability: The probability of occurence of (label, pixel, bin) = (i, j, k)
        lfrequency: The number of occurence of label i
        lcount: The total number of labels
        lprobability: The probability of occurence of label i
    """
    def __init__(self):
        # 10 for labels: 0 ~ 9
        # 784 for features: 28x28 pixels
        # 32 for bins: make "gray level 256" to 32 bins. That is, each bin has 8 gray levels 
        self.pfrequecy = np.zeros(shape=(10, 784, 32), dtype=int)
        self.pprobability = np.zeros(shape=(10, 784, 32), dtype=float)
        self.lfrequency = np.zeros(shape=(10), dtype=int)
        self.lcount = 0
        self.lprobability = np.zeros(shape=(10), dtype=float)

    def train(self, Xtrain: Image, Ytrain: Lable):
        nImage, nPixel = Xtrain.nImage, (Xtrain.nrow * Xtrain.ncol)
        self.lcount = Ytrain.nLable
        # Tally the values of each pixel into 32 bins
        bins = np.floor(Xtrain.Images / 8).reshape(nImage, nPixel).astype(int)

        # Update the frequency, count for pixel and label
        for idxImage in range(nImage):
            curlable = Ytrain.Lables[idxImage]
            self.lfrequency[curlable] += 1

            for idxPixel in range(nPixel):
                self.pfrequecy[curlable][idxPixel][bins[idxImage][idxPixel]] += 1

        # Calculate the probability of each lable
        self.lprobability = self.lfrequency / self.lcount
        
        # Laplace Smoothing
        self.pfrequecy += 1  

        for idxlabel in range(10):
            # Calculate the probability of occurence of (label, pixel, bin) = (i, j, k)
            self.pprobability[idxlabel]= self.pfrequecy[idxlabel]/ (self.lfrequency[idxlabel]+32)

    def predict(self, Xtest: Image):
        nImage, nPixel = Xtest.nImage, (Xtest.nrow * Xtest.ncol)
        predictY = np.zeros(shape=(nImage), dtype=int)
        posterior = np.zeros(shape=(nImage, 10), dtype=float)
        bins = np.floor(Xtest.Images / 8).reshape(nImage, nPixel).astype(int)
        idx = np.arange(nPixel)

        for idxImage in range(nImage):
            for idxLable in range(10):
                # calculate likelohood, prior and posterior in log scale
                likelihood = np.sum(np.log(self.pprobability[idxLable, idx, bins[idxImage][idx]]))
                prior = np.log(self.lprobability[idxLable])
                posterior[idxImage][idxLable] = likelihood + prior

            predictY[idxImage] = np.argmax(posterior[idxImage])
            # marginalize posterior to let them sum up to 1
            posterior[idxImage] /= np.sum(posterior[idxImage])

        return predictY, posterior

    def calError(self, Ypredict, Ytest: Lable):
        nLable = Ytest.nLable
        wrong = np.sum(Ypredict != Ytest.Lables)
        return wrong / nLable

    def Imagination(self):
        imagination = np.zeros(shape=(10, 784), dtype=np.uint8)
        for idxLable in range(10):
            for idxPixel in range(784):
                imagination[idxLable][idxPixel] = np.argmax([np.sum(self.pfrequecy[idxLable][idxPixel][:16]), np.sum(self.pfrequecy[idxLable][idxPixel][16:])])

        print("Imagination of numbers in Bayesian classifier:\n")
        for idxLabel in range(10):
            print(idxLabel, ":")
            print(imagination[idxLabel].reshape(28, 28), "\n")
        
class ContinuousNBClassfier:
    """Initialize a Naive Bayes Classifier for Continuous Mode

    Variables:
        psum: The sum of gray level of (label, pixel) = (i, j)
        psquaresum: The square sum of gray level of (label, pixel) = (i, j)
        pmean: The mean of gray level of (label, pixel) = (i, j)
        pvariance: The variance of gray level of (label, pixel) = (i, j)
        lfrequency: The number of occurence of label i
        lcount: The total number of labels
        lprobability: The probability of occurence of label i
    """
    def __init__(self):
        # 10 for labels: 0 ~ 9
        # 784 for features: 28x28 pixels
        self.psum = np.zeros(shape=(10, 784), dtype=float)
        self.psquaresum = np.zeros(shape=(10, 784), dtype=float)
        self.pmean = np.zeros(shape=(10, 784), dtype=float)
        self.pvariance = np.zeros(shape=(10, 784), dtype=float)
        self.lfrequency = np.zeros(shape=(10), dtype=int)
        self.lcount = 0
        self.lprobability = np.zeros(shape=(10), dtype=float)        

    def train(self, Xtrain: Image, Ytrain: Lable, smoothing = 1000):
        nImage, nPixel = Xtrain.nImage, (Xtrain.nrow * Xtrain.ncol)
        self.lcount = Ytrain.nLable
        data = Xtrain.Images.reshape(nImage, nPixel).astype(float)

        # Update the sum, square sum of pixel and the frequency of label
        for idxImage in range(nImage):
            curlable = Ytrain.Lables[idxImage]
            self.lfrequency[curlable] += 1
            self.psum[curlable] += data[idxImage]
            self.psquaresum[curlable] += data[idxImage] ** 2

        # Calculate the probability of each lable
        self.lprobability = self.lfrequency / self.lcount

        # Calculate the mean(MLE) and variance(MLE) of pixel
        for idxlabel in range(10):
            self.pmean[idxlabel] = self.psum[idxlabel] / self.lfrequency[idxlabel]
            # Var[x] = E[x^2] - (E[x])^2
            self.pvariance[idxlabel] = self.psquaresum[idxlabel] / self.lfrequency[idxlabel] - self.pmean[idxlabel] ** 2 + smoothing

    def predict(self, Xtest: Image):
        nImage, nPixel = Xtest.nImage, (Xtest.nrow * Xtest.ncol)
        data = Xtest.Images.reshape(nImage, nPixel).astype(float)
        predictY = np.zeros(shape=(nImage), dtype=int)
        posterior = np.zeros(shape=(nImage, 10), dtype=float)

        for idxImage in range(nImage):
            for idxLable in range(10):
                # calculate likelohood, prior and posterior in log scale
                # probability of Gaussian distribution in log scale = -log(2 * pi * variance)/2 - (x-mean)^2 / (2*variance)
                likelihood = -np.log(2 * np.pi * self.pvariance[idxLable])/2 - ((data[idxImage] - self.pmean[idxLable]) ** 2) / (2 * self.pvariance[idxLable])
                prior = np.log(self.lprobability[idxLable])                
                posterior[idxImage][idxLable] = np.sum(likelihood) + prior

            # marginalize posterior to let them sum up to 1
            posterior[idxImage] /= np.sum(posterior[idxImage])
            # the minimum posterior in log scale is the most probable
            predictY[idxImage] = np.argmin(posterior[idxImage])

        return predictY, posterior

    def calError(self, Ypredict, Ytest: Lable):
        nLable = Ytest.nLable
        wrong = np.sum(Ypredict != Ytest.Lables)
        return wrong / nLable

    def Imagination(self):
        imagination = (self.pmean >= 128).reshape(10, 28, 28).astype(np.uint8)
        print("Imagination of numbers in Bayesian classifier:\n")
        for idxLabel in range(10):
            print(idxLabel, ":")
            print(imagination[idxLabel], "\n")
        
def main(Args):
    ### Preprocessing ###
    toggle = Args.toggle
    Xtrain = readImage(TrainImagePath)
    Xtest = readImage(TestImagePath)
    Ytrain = readLable(TrainLablePath)
    Ytest = readLable(TestLablePath)
    ### Preprocessing END ###

    # Discrete Mode
    if toggle == 0:
        NBC = DiscreteNBClassifer()
    # Continuous mode
    elif toggle == 1:
        NBC = ContinuousNBClassfier()

    ### Main Part ###
    NBC.train(Xtrain, Ytrain)
    predictY, posterior = NBC.predict(Xtest)
    errorRate = NBC.calError(predictY, Ytest)

    ### Print the Result ###
    for cases in range(Ytest.nLable):
        print("Posterior (in log scale):")
        for label in range(10):
            print(label, ": ", posterior[cases][label], sep="")
        print("Prediction: ", predictY[cases], ", Ans: ", Ytest.Lables[cases], "\n", sep="") 
    NBC.Imagination()
    print("Error rate:", errorRate)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('toggle', type=int, help="Please enter toggle option: 1 for discrete; 2 for continuous")
    args = parser.parse_args()
    main(args)