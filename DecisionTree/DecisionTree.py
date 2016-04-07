# -*- coding: utf-8 -*-
from math import log2
from DrawTree import createPlot


def calEntropy(dataSet):
    categories = {}
    sampleCnt = len(dataSet)
    ent = 0
    if sampleCnt == 0:
        return ent

    for dataLine in dataSet:
        if dataLine[-1] in categories:
            categories[dataLine[-1]] += 1
        else:
            categories[dataLine[-1]] = 1

    for cnt in categories.values():
        p = float(cnt) / sampleCnt
        ent += p * log2(p)
    return - ent


def calGini(dataSet):
    samples = [data[-1] for data in dataSet]
    numSamples = len(samples)
    sampleCnt = {}
    for smp in samples:
        if smp not in sampleCnt:
            sampleCnt[smp] = 0
        sampleCnt[smp] += 1
    GiniValue = 1
    for value in sampleCnt.values():
        GiniValue -= (value / numSamples) ** 2

    return GiniValue


def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedVec = featVec[:axis]
            reducedVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedVec)
    return retDataSet


def chooseBestFeatureWithGini(dataSet):
    if len(dataSet) == 0:
        return None
    bestFeature = -1
    smallestGiniValue = 1
    dataSetLen = len(dataSet)

    for i in range(len(dataSet[0]) - 1):

        GiniValue = 0
        attr = set([dataSet[x][i] for x in range(len(dataSet))])

        for value in attr:
            subDataSet = splitDataSet(dataSet, i, value)
            GiniValue += len(subDataSet) / dataSetLen * calGini(subDataSet)

        if GiniValue - smallestGiniValue < -1e-5:
            bestFeature = i
            smallestGiniValue = GiniValue

        print("i = %d, infoGini = %f" % (i, GiniValue))

    if bestFeature == -1:
        return None
    else:
        return bestFeature


def chooseBestFeatureWithEntropy(dataSet):
    if len(dataSet) == 0:
        return None
    bestFeature = -1
    largestGain = 0
    baseInfoGain = calEntropy(dataSet)
    dataSetLen = len(dataSet)

    attrDic = {}

    for i in range(len(dataSet[0]) - 1):
        attrDic[i] = set([dataSet[x][i] for x in range(len(dataSet))])
        # print(attrDic)

    for i in range(len(dataSet[0]) - 1):

        infoGain = baseInfoGain
        attr = attrDic[i]
        for value in attr:
            subDataSet = splitDataSet(dataSet, i, value)
            infoGain -= float(len(subDataSet)) / dataSetLen * calEntropy(subDataSet)

        if infoGain - largestGain > 1e-5:
            # print('****>>>', infoGain, '   ',largestGain)
            bestFeature = i
            largestGain = infoGain

        print("i = %d, infoGain = %f" % (i, infoGain))
    # print(bestFeature)
    if bestFeature == -1:
        return None
    else:
        return bestFeature


def calMostExamplesClass(examples):
    cnt = {}
    for example in examples:
        if example in cnt:
            cnt[example] += 1
        else:
            cnt[example] = 1
    lst = sorted(cnt.items(), key=lambda x: x[1], reverse=True)
    return lst[0][0]


class TreeNode:
    def __init__(self, feature, isLeaf=False, fatherInfo=(None, None)):
        self.children = {}
        self.feature = feature
        self.isLeaf = isLeaf
        self.father = fatherInfo[0]
        self.path = fatherInfo[1]


class TreeError(BaseException):
    def __init__(self, info):
        self.message = info


class DecisionTree:
    def __init__(self, dataSet, labels, method='entropy'):
        if method == 'entropy':
            self.chooseFunc = chooseBestFeatureWithEntropy
        elif method == 'gini':
            self.chooseFunc = chooseBestFeatureWithGini
        else:
            raise TreeError("Unsupported method!")

        self.labels = labels[:]
        self.labelIndex = {}

        for i, label in enumerate(self.labels):
            if label in self.labelIndex:
                raise TreeError('Duplicated labels %s' % label)
            self.labelIndex[label] = i

        self.attrValues = []

        for i in range(len(dataSet[0]) - 1):
            self.attrValues.append(list(set([val[i] for val in dataSet])))
        self.root = self.createTree(dataSet, labels, self.attrValues[:])

    def predict(self, attrValues):
        root = self.root
        while not root.isLeaf:
            attr = root.feature
            value = attrValues[self.labelIndex[attr]]
            if value not in root.children:
                raise TreeError('Unknown attr value: %s' % value)
            root = root.children[value]
        return root.feature

    def createTree(self, dataSet, labels, attrValues, fatherInfo=(None, None)):
        examples = [example[-1] for example in dataSet]

        if examples.count(examples[0]) == len(examples):
            return TreeNode(examples[0], True, fatherInfo)

        if len(labels) == 0 or len(set([''.join(x[:-1]) for x in dataSet])) == 1:
            return TreeNode(calMostExamplesClass(examples), True, fatherInfo)

        featIndex = self.chooseFunc(dataSet)
        feature = labels[featIndex]

        retNode = TreeNode(feature, fatherInfo=fatherInfo)

        new_labels = labels[:featIndex]
        new_labels.extend(labels[featIndex + 1:])

        new_attrValues = attrValues[:featIndex]
        new_attrValues.extend(attrValues[featIndex + 1:])
        # attrValues = set([data[featIndex] for data in dataSet])

        for value in attrValues[featIndex]:
            subDataSet = splitDataSet(dataSet, featIndex, value)
            if len(subDataSet) == 0:
                retNode.children[value] = TreeNode(calMostExamplesClass(examples), True, (retNode, value))
            else:
                retNode.children[value] = self.createTree(subDataSet, new_labels, new_attrValues, (retNode, value))
        return retNode

    def pruning(self, testDataSet):
        nodeLst = breadthTravel(self.root)
        i = len(nodeLst) - 1
        while i >= 0:
            currentAccuracy = self.calAccuracy(testDataSet)
            testNode = nodeLst[i]
            pathDic = {}
            travelNode = testNode
            while travelNode.father != None:
                pathDic[travelNode.father.feature] = travelNode.path
                travelNode = travelNode.father
            filteredData = testDataSet[:]
            for key, value in pathDic.items():
                filteredData = [line for line in filteredData if line[self.labelIndex[key]] == value]
            # 算最多的分类
            examples = [example[-1] for example in filteredData]
            newNode = TreeNode(calMostExamplesClass(examples), True, (testNode.father, testNode.path))

            if testNode.father != None:
                testNode.father.children[testNode.path] = newNode
            else:
                self.root = newNode

            newAccuracy = self.calAccuracy(testDataSet)

            if newAccuracy - currentAccuracy < 1e-5:
                if testNode.father != None:
                    testNode.father.children[testNode.path] = testNode
                else:
                    self.root = testNode
            i -= 1

    def calAccuracy(self, testDataSet):
        predictResult = [result for result in map(self.predict, testDataSet)]
        groundTruth = [GT[-1] for GT in testDataSet]
        difference = 0

        for i in range(len(predictResult)):
            if predictResult[i] == groundTruth[i]:
                difference += 1

        return difference / len(predictResult)


def breadthTravel(DT):
    retLst = []
    retLst.append(DT)
    if DT.isLeaf:
        return retLst
    n = 0
    while True:
        for item in retLst[n].children.values():
            if not item.isLeaf:
                retLst.append(item)
        n += 1
        if n == len(retLst):
            break
    return retLst


def preOrder(DT):
    print(DT.feature, DT.children.keys())
    if DT.father is not None:
        print(DT.father.feature, DT.path, end='\n\n')
    if DT.isLeaf:
        return
    for item in DT.children.values():
        preOrder(item)


def __loadTrainData(dataSetPath):
    f = open(dataSetPath)
    s = f.readline()
    labels = []
    labels.extend(s.strip().split('\t'))

    dataSet = __loadTestData(f)

    return dataSet, labels


def __loadTestData(f):
    file = None
    file = f
    if isinstance(f, str):
        file = open(f)

    dataSet = []
    for line in file:
        dataSet.append(line.strip().split())

    return dataSet


if __name__ == '__main__':
    dataSet, labels = __loadTrainData('4_2_train.txt')
    # print (labels)
    dt = DecisionTree(dataSet, labels)
    testSet = __loadTestData('4_2_test.txt')
    print(testSet)
    print('Accuracy before pruning:', dt.calAccuracy(testSet))
    dt.pruning(testSet)
    print('Accuracy after pruning:', dt.calAccuracy(testSet))
    createPlot(dt.root)
