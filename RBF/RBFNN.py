import math
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt


def activeFunction(x, parameters):
    center, delta = parameters
    e = np.e
    base = ((x - center) ** 2).sum()
    return e ** (-(base ** 2) / (2 * (delta ** 2)))


class HNode(object):
    def __init__(self, center, delta, af=activeFunction):
        self.parameters = [center.copy(), delta]
        self.af = af

    def calOutput(self, x):
        return self.af(x, self.parameters)


class ONode(object):
    def __init__(self, inputNum):
        self.weights = []
        for i in range(inputNum):
            self.weights.append(np.random.random())

    def calOutput(self, input):
        return reduce(lambda x, y: x + y, map(lambda x, y: x * y, input, self.weights))


class RBFNN(object):
    def __init__(self, outputNum, centers, rate=0.2):
        self.hiddenLayer = []
        self.outputLayer = []
        self.inputNum = len(centers[0])
        self.rate = rate
        hiddenNum = len(centers)

        center_pairs = [[x, y] for x in centers for y in centers]
        max_dist = np.linalg.norm(max(center_pairs, key=lambda k: np.linalg.norm(k[0] - k[1])))
        delta_init = max_dist / np.sqrt(2 * hiddenNum)

        for i in range(hiddenNum):
            self.hiddenLayer.append(HNode(centers[i], delta_init))

        for i in range(outputNum):
            self.outputLayer.append(ONode(hiddenNum))

    def calHiddenLayerOutput(self, input):
        return [hiddenNode.calOutput(input) for hiddenNode in self.hiddenLayer]

    def calOutLayerOutput(self, input):
        return [outputNode.calOutput(input) for outputNode in self.outputLayer]

    def train(self, trainData, trainLabel, time=1000):
        n = 0
        while not n == time:
            n += 1
            if n % 100 == 0:
                print(n)

            k = 0
            for i, train_data in enumerate(trainData):
                print(k)
                k += 1
                true_value = trainLabel[i]
                hiddenLayerOutput = self.calHiddenLayerOutput(train_data)
                outLayerOutput = self.calOutLayerOutput(hiddenLayerOutput)
                _C_update = []
                _delta_update = []
                _weight_update = []
                for i, hiddenNode in enumerate(self.hiddenLayer):
                    center = hiddenNode.parameters[0]
                    delta = hiddenNode.parameters[1]
                    G = hiddenLayerOutput[i]
                    C_para = self.rate * G * (train_data - center) / (delta ** 2)
                    delta_para = self.rate * G * (((train_data - center) ** 2).sum()) / (delta ** 3)
                    weight_para = self.rate * G
                    C_delta_e = reduce(lambda x, y: x + y, [outputNode.weights[i] * (true_value[j] - outLayerOutput[j])
                                                            for j, outputNode in enumerate(self.outputLayer)])

                    _weight_update.append([weight_para * (true_value[j] - outLayerOutput[j])
                                           for j in range(len(self.outputLayer))])

                    _C_update.append(C_para * C_delta_e)
                    _delta_update.append(delta_para * C_delta_e)

                for i, hiddenNode in enumerate(self.hiddenLayer):
                    hiddenNode.parameters[0] += _C_update[i]
                    hiddenNode.parameters[1] += _delta_update[i]
                    for j, outputNode in enumerate(self.outputLayer):
                        outputNode.weights[i] += _weight_update[i][j]

    def predict(self, x):
        return self.calOutLayerOutput(self.calHiddenLayerOutput(x))


if __name__ == '__main__':
    centers = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
    n = RBFNN(1, centers, 0.02)
    trainData = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
    label = np.array([[0], [1], [1], [1]], dtype=np.float64)
    n.train(trainData, label, 10000)

    while True:
        s = input()
        s = s.strip().split(' ')
        s = np.array([float(num) for num in s])
        print(n.predict(s))
