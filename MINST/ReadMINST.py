import struct
import numpy as np
import sys

sys.path.append('..\\')
from RBF import RBFNN
from functools import reduce
import matplotlib.pyplot as plt


def getNextImage(fileName):
    image_file = open(fileName, 'rb')
    buf = image_file.read()
    index = 0
    magic, numImages, numRows, numColumns = struct.unpack_from('>IIII', buf, index)
    index += struct.calcsize('>IIII')
    image_cnt = 0
    while image_cnt != numImages:
        image_cnt += 1
        im = struct.unpack_from('>%dB' % (numRows * numColumns,), buf, index)
        index += struct.calcsize('>%dB' % (numRows * numColumns,))
        yield np.array(im, dtype=np.float64)


def getNextLabels(fileName):
    labels_file = open(fileName, 'rb')
    buf = labels_file.read()
    index = 0
    magicNum, numLabels = struct.unpack_from('>II', buf, index)
    index += struct.calcsize('>II')
    label_cnt = 0
    while label_cnt != numLabels:
        label_cnt += 1
        label = struct.unpack_from('1B', buf, index)
        index += 1
        yield label[0]


def readImages(imageFile, labelsFile):
    imagesDic = {key: [] for key in range(10)}
    for image, label in zip(getNextImage(imageFile), getNextLabels(labelsFile)):
        imagesDic[label].append(image)
    return imagesDic


if __name__ == '__main__':
    imagesDic = readImages('train-images.idx3-ubyte', 'train-labels.idx1-ubyte')
    labelLst = []

    trainData = []
    trainLabel = []
    centers = []

    for label, values in imagesDic.items():
        centers.extend([value[:] for value in values[::1000]])
    for label in imagesDic.keys():
        trainData.extend(imagesDic[label])
        trainLabel.extend(len(imagesDic[label]) * [[label]])

    trainData = np.array(trainData, dtype=np.float64)
    trainLabel = np.array(trainLabel, dtype=np.float64)

    print(trainData.shape)
    print(trainLabel.shape)

    centers = np.array(centers, dtype=np.float64)

    rbf = RBFNN.RBFNN(1, centers)

    rbf.train(trainData, trainLabel)

    print(rbf.predict(trainData[1024][0]))
    print(trainData[1024][1])
    print(rbf.predict(trainData[2025][0]))
    print(trainData[2025][1])
    print(rbf.predict(trainData[3026][0]))
    print(trainData[3026][1])
    print(rbf.predict(trainData[1027][0]))
    print(trainData[1027][1])
    print(rbf.predict(trainData[1028][0]))
    print(trainData[1028][1])
