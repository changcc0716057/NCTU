import gzip
import copy as cp
import numpy as np

TrainImagePath = 'C:/Users/jackson/Desktop/chiacheng/visualstudiocode/nctu110fall/ML/hw2/train-images-idx3-ubyte.gz'
TrainLabelPath = 'C:/Users/jackson/Desktop/chiacheng/visualstudiocode/nctu110fall/ML/hw2/train-labels-idx1-ubyte.gz'
np.set_printoptions(threshold=np.inf)

def readImage(path):
    with gzip.open(path, 'r') as f:
        # first 16 bytes: 4 for magic number, 4 for number of Images, 4 for number of rows, and 4 for number of columns
        mNumber = int.from_bytes(bytes=f.read(4), byteorder='big', signed=False)
        nImage = int.from_bytes(bytes=f.read(4), byteorder='big', signed=False)
        nrow = int.from_bytes(bytes=f.read(4), byteorder='big', signed=False)
        ncol = int.from_bytes(bytes=f.read(4), byteorder='big', signed=False)

        # other bytes: pixels of Image (each pixel is an unsigned byte)
        Images = np.frombuffer(f.read(), dtype=np.uint8).reshape(nImage, nrow * ncol)
        Images_2bin = cp.deepcopy(Images)
        Images_2bin[Images_2bin < 128] = 0
        Images_2bin[Images_2bin >= 128] = 1
        Xdata = Image(mNumber, nImage, nrow, ncol, Images, Images_2bin.astype(int))
    return Xdata

def readLabel(path):
    with gzip.open(path, 'r') as f:
        # first 8 bytes: 4 for magic number, 4 for number of Labels
        mNumber = int.from_bytes(bytes=f.read(4), byteorder='big', signed=False)
        nLabel = int.from_bytes(bytes=f.read(4), byteorder='big', signed=False)

        # other bytes: Labels (each Label is an unsigned byte)
        Labels = np.frombuffer(f.read(), dtype=np.uint8)
        Ydata = Label(mNumber, nLabel, Labels)
    return Ydata

class Image:
    def __init__(self, mNumber, nImage, nrow, ncol, Images, Images_2bin):
        self.MagicNumber = mNumber
        self.nImage = nImage
        self.nrow = nrow
        self.ncol = ncol
        self.Images = Images
        self.Images_2bin = Images_2bin

class Label:
    def __init__(self, mNumber, nLabel, Labels):
        self.MagicNumber = mNumber
        self.nLabel = nLabel
        self.Labels = Labels

class EMmodel:
    """
    Initialize a EM model.
    Variables:
        Xtrain: The data used to train
        Ytrain: The label only used to figure out which class belong to which number
        nPixel: The number of pixels per image
        nData: The number of datas in Xtrain and Ytrain
        plabel: The probability of we choose which label. Initially, the probability is 0.1 for label 0~9
        ppixel: The probability of the specified pixel is 1, eX: p(i,j) is the jth pixel for label i in current image. Initially, all the
                probability is 0.5.
        confusionMat: The confusion matrix of the model
        classlabel: The label of the class  
    """
    def __init__(self, Xtrain: Image, Ytrain: Label):
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        self.nPixel = Xtrain.nrow * Xtrain.ncol
        self.nData = Xtrain.nImage
        # Randomly initialize plabel and ppixel 
        self.plabel = np.random.rand(10).astype(np.float_)
        self.plabel = self.plabel / np.sum(self.plabel)
        self.ppixel = np.random.rand(10, self.nPixel).astype(np.float_)
        self.assigntable = np.zeros(shape=(10, 10), dtype=np.int)
        self.confusionMat = np.zeros(shape=(10, 2, 2), dtype=np.int)
        self.classlabel = np.full(10, -1)

    def Estep(self):
        # Calculate the likelihood of per pixel
        likelihood_pixel = np.zeros(shape=(10, self.nData, self.nPixel), dtype=np.float_)
        # for each label and each pixel
        for label in range(10):
            for pixel in range(self.nPixel):
                likelihood_pixel[label, :, pixel] = self.ppixel[label][pixel] * self.Xtrain.Images_2bin[:, pixel] + (1 - self.ppixel[label][pixel]) * (1 - self.Xtrain.Images_2bin[:, pixel])
        # Calculate the likelihood of per data
        likelihood = np.zeros(shape=(10, self.nData), dtype=np.float_)
        for label in range(10):
            likelihood[label] = np.prod(likelihood_pixel[label], axis=1) * self.plabel[label]

        # To avoid the sum of weights is 0, let the smoothing value be the minimum nonzero value in likelihood
        nonzerolikelihood = np.ma.masked_array(likelihood, mask=(likelihood==0))
        smoothing = nonzerolikelihood.min()
        weight = (likelihood + smoothing) / (np.sum(likelihood, axis=0) + smoothing * 10)

        return weight

    def Mstep(self, weight):
        plabel_new = np.zeros(shape=(10), dtype=np.float_)
        ppixel_new = np.zeros(shape=(10, self.nPixel), dtype=np.float_)

        weightsum = np.sum(weight, axis=1)
        # update the probability of pixel
        for label in range(10):
            for pixel in range(self.nPixel):
                ppixel_new[label][pixel] = np.sum(weight[label, :] * self.Xtrain.Images_2bin[:, pixel]) / weightsum[label]

        # update the probability of label
        plabel_new = weightsum / np.sum(weightsum)
        return plabel_new, ppixel_new

    def norm(self, ppixel_new):
        return np.sum(abs(ppixel_new - self.ppixel))

    def train(self):
        ExitCode = False
        cnt = 0
        while not ExitCode:
            weight = self.Estep()
            plabel_new, ppixel_new = self.Mstep(weight)
            diff = self.norm(ppixel_new)
            if cnt >= 50 or diff <= 15:
                ExitCode = True
            self.plabel = plabel_new
            self.ppixel = ppixel_new
            cnt += 1
            for i in range(10):
                self.imagination(i, self.ppixel[i])
            print("No. of Iteration: ", cnt, ", Difference: ", diff, sep="")
            print("\n-------------------------------------------------------------------\n")
        return cnt
            
    def imagination(self, Numclass, image, final=False):
        imgName = "labeled class " if final else "class "
        print(imgName, Numclass, ":", sep="")
        image = image.reshape(28,28)
        for row in range(28):
            for col in range(28):
                val = (image[row][col] >= 0.5) * 1
                print(val, end=" ")
            print()
        print()

    def AssignClassLabel(self):
        likelihood_pixel = np.zeros(shape=(10, self.nData, self.nPixel), dtype=np.float_)
        for label in range(10):
            for pixel in range(self.nPixel):
                likelihood_pixel[label, :, pixel] = (self.ppixel[label][pixel] * self.Xtrain.Images_2bin[:, pixel] + (1 - self.ppixel[label][pixel]) * (1 - self.Xtrain.Images_2bin[:, pixel])) * (10 ** (1/3))
        
        # Calculate the likelihood of per data
        likelihood = np.zeros(shape=(10, self.nData), dtype=np.float_)
        for label in range(10):
            likelihood[label] = np.prod(likelihood_pixel[label], axis=1) * self.plabel[label]
        
        maxlikelihood = np.argmax(likelihood, axis=0)
        for idx in range(self.nData):
            self.assigntable[self.Ytrain.Labels[idx]][maxlikelihood[idx]] += 1

        tmptable = cp.deepcopy(self.assigntable)
        for label in range(10):
            maximum = np.argwhere(tmptable==tmptable.max()).flatten()
            self.classlabel[maximum[0]] = maximum[1]
            tmptable[maximum[0], :] = -1
            tmptable[:, maximum[1]] = -1

    def constructConfusionTable(self):
        for label in range(10):
            row, col = label, self.classlabel[label]
            self.confusionMat[label][0][0] = self.assigntable[row][col]
            self.confusionMat[label][0][1] = np.sum(self.assigntable[row, :]) - self.assigntable[row][col]
            self.confusionMat[label][1][0] = np.sum(self.assigntable[:, col]) - self.assigntable[row][col]
            self.confusionMat[label][1][1] = np.sum(self.assigntable) - self.confusionMat[label][0][0] - self.confusionMat[label][0][1] - self.confusionMat[label][1][0]

    def printConfusionTable(self, Numclass):
        print("Confusion Matrix ", Numclass, ":", sep="")
        print("                Predict number ", Numclass, " Predict not number ", Numclass, sep="")
        print("Is number ", Numclass, "     ", '{:>9}'.format(str(self.confusionMat[Numclass][0][0])), "        ", '{:>12}'.format(str(self.confusionMat[Numclass][0][1])), sep="")
        print("Isn\'t number ", Numclass, "  ", '{:>9}'.format(str(self.confusionMat[Numclass][1][0])), "        ", '{:>12}'.format(str(self.confusionMat[Numclass][1][1])), sep="")
        print("\nSensitivity (Successfully predict number ", Numclass, ")    : ", self.confusionMat[Numclass][0][0] / np.sum(self.confusionMat[Numclass][0, :]), sep="")
        print("Specificity (Successfully predict not number ", Numclass, "): ", self.confusionMat[Numclass][1][1] / np.sum(self.confusionMat[Numclass][1, :]), sep="")
        print("\n-------------------------------------------------------------------\n")

def main():
    ### Preprocessing ###
    Xtrain = readImage(TrainImagePath)
    Ytrain = readLabel(TrainLabelPath)
    ### Preprocessing END ###
    em = EMmodel(Xtrain, Ytrain)
    NumOfIter = em.train()
    em.AssignClassLabel()
    em.constructConfusionTable()
    print("-------------------------------------------------------------------")
    for label in range(10):
        em.imagination(label, em.ppixel[em.classlabel[label]], True)
    print("===================================================================\n")
    for label in range(10):
        em.printConfusionTable(label)
    print("Total iteration to converge: ", NumOfIter, sep="")
    print("Total error rate: ", (em.nData - np.sum(em.confusionMat[:, 0, 0])) / em.nData, sep="")

if __name__ == "__main__":
    main()