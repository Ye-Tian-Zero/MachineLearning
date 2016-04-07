from math import e
from random import random


def sigmoid(x):
    return 1 / (1 + e ** (-x))


class Neuron(object):
    def __init__(self, numInput, af=sigmoid):
        self.weights = []
        for i in range(numInput):
            self.weights.append(random())
        self.threshold = random()
        self.activationFunction = af

    def output(self, input):
        alpha = 0
        for i, value in enumerate(input):
            alpha += value * self.weights[i]

        return self.activationFunction(alpha - self.threshold)


class BPNeuronNetwork(object):
    def __init__(self, numInputLayer, numHiddenLayer, numOutputLayer, rateH=0.2, rateO=0.2):
        self.hiddenLayer = []
        self.outputLayer = []
        for i in range(numHiddenLayer):
            self.hiddenLayer.append(Neuron(numInputLayer))
        for i in range(numOutputLayer):
            self.outputLayer.append(Neuron(numHiddenLayer))
        self.rateH = rateH
        self.rateO = rateO

    def predict(self, input):
        hiddenLayerOutput = self.calLayerOutput(self.hiddenLayer, input)
        return self.calLayerOutput(self.outputLayer, hiddenLayerOutput)

    def calLayerOutput(self, layer, input):
        LayerOutput = []
        for neuron in layer:
            LayerOutput.append(neuron.output(input))
        return LayerOutput

    def train(self, trainData, train_times=10000):
        Cnt = 0
        while True:
            if Cnt == train_times:
                break
            Cnt += 1
            if Cnt % 1000 == 0:
                print(Cnt)
            for dataSample in trainData:
                hiddenLayerOutput = self.calLayerOutput(self.hiddenLayer, dataSample[:-1])
                outputLayerOutput = self.calLayerOutput(self.outputLayer, hiddenLayerOutput)
                groundTruth = dataSample[-1]

                g = []
                E = []

                for j, y in enumerate(groundTruth):
                    y_ = outputLayerOutput[j]
                    g.append(y_ * (1 - y_) * (y - y_))

                for h, H_neuron in enumerate(self.hiddenLayer):
                    sum = 0
                    for j, O_neuron in enumerate(self.outputLayer):
                        sum += O_neuron.weights[h] * g[j]
                    E.append(hiddenLayerOutput[h] * (1 - hiddenLayerOutput[h]) * sum)

                for h, H_neuron in enumerate(self.hiddenLayer):
                    for i in range(len(H_neuron.weights)):
                        H_neuron.weights[i] += self.rateH * E[h] * dataSample[i]
                    H_neuron.threshold -= self.rateH * E[h]

                for j, neuron in enumerate(self.outputLayer):
                    for h in range(len(neuron.weights)):
                        neuron.weights[h] += self.rateO * g[j] * hiddenLayerOutput[h]
                    neuron.threshold -= self.rateO * g[j]


if __name__ == '__main__':
    n = BPNeuronNetwork(2, 10, 1)

    n.train([[0, 1, [1]],
             [1, 0, [1]],
             [0, 0, [0]],
             [1, 1, [0]]])

    while True:
        testNum = input("input 2 nums: ")
        testNum = testNum.strip().split(' ')
        print(n.predict([int(testNum[0]), int(testNum[1])]))
