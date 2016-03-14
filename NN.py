from math import e


def sigmoid(x):
    return 1 / (1 + e ** (-x))


class Neuron(object):
    def __init__(self, numInput, weight, af=sigmoid):
        self.weights = [weight] * numInput
        self.threshold = 0
        self.activationFunction = af

    def output(self, input):
        alpha = 0
        for i, value in enumerate(input):
            alpha += value * self.weights[i]

        return self.activationFunction(alpha - self.threshold)


class BPNeuronNetwork(object):
    def __init__(self, numInputLayer, numHiddenLayer, numOutputLayer, weight=1.0, rateH=0.5, rateO=0.5):
        self.hiddenLayer = [Neuron(numInputLayer, weight)] * numHiddenLayer
        self.outputLayer = [Neuron(numHiddenLayer, weight)] * numOutputLayer
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

    def train(self, trainData):
        Cnt = 0
        while True:
            if Cnt == 10000:
                break
            Cnt += 1
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
                        neuron.weights[h] += self.rateH * g[j] * hiddenLayerOutput[h]
                    neuron.threshold -= self.rateH * g[j]


if __name__ == '__main__':
    n = BPNeuronNetwork(2, 1, 1)

    n.train([[0, 0, [1]],
             [0, 1, [1]],
             [1, 0, [0]],
             [1, 1, [0]]])

    while True:
        testNum = input("input 2 nums: ")
        testNum = testNum.strip().split(' ')
        print(n.predict([int(testNum[0]), int(testNum[1])]))