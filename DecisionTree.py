# -*- coding: utf-8 -*-
from math import log2

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
        p = cnt / sampleCnt
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
            reducedVec.extend(featVec[axis + 1 :])
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
             
        if GiniValue < smallestGiniValue:
            bestFeature = i
            smallestGiniValue = GiniValue
            
        print ("i = %d, infoGini = %f" % (i, GiniValue))
        
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
    for i in range(len(dataSet[0]) - 1):
        
        attr = set([dataSet[x][i] for x in range(len(dataSet))])
        infoGain = baseInfoGain
        
        for value in attr:
            subDataSet = splitDataSet(dataSet, i, value)
            infoGain -= len(subDataSet) / dataSetLen \
            * calEntropy(subDataSet)
             
        if infoGain > largestGain:
            bestFeature = i
            largestGain = infoGain
            
        print ("i = %d, infoGain = %f" % (i, infoGain))
        
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
    lst = sorted(cnt.items(), key = lambda x : x[1], reverse = True)
    return lst[0][0]
          
class TreeNode:
    def __init__(self, feature, isleaf = False):
        self.children = {}
        self.feature = feature
        self.isleaf = isleaf

class TreeError(BaseException):
    def __init__(self, info):
        self.message = info
  
class DecisionTree:
    
    def __init__(self, dataSet, labels, method = 'entropy'):
        if method == 'entropy':
            self.chooseFunc = chooseBestFeatureWithEntropy
        elif method == 'gini':
            self.chooseFunc = chooseBestFeatureWithGini
        else:
            raise TreeError("Unsupported method!")
            
        self.attrValues = []
        for i in range(len(dataSet[0]) - 1):
            self.attrValues.append(list(set([val[i] for val in dataSet])))
        self.root = self.createTree(dataSet, labels, self.attrValues)
        
        
    def createTree(self, dataSet, labels, attrValues):
            
        examples = [example[-1] for example in dataSet]
        
        if examples.count(examples[0]) == len(examples):
            return TreeNode(examples[0], True)
        
        if len(labels) == 0 or len(set([''.join(x[:-1]) for x in dataSet])) == 1:
            return TreeNode(calMostExamplesClass(examples), True)
        
        featIndex = self.chooseFunc(dataSet)
        feature = labels[featIndex]
        
        retNode = TreeNode(feature)
        
        new_labels = labels[:featIndex]
        new_labels.extend(labels[featIndex + 1 :])
        
        new_attrValues = attrValues[:featIndex]
        new_attrValues.extend(attrValues[featIndex + 1 :])
        #attrValues = set([data[featIndex] for data in dataSet])
      
        for value in attrValues[featIndex]:
            subDataSet = splitDataSet(dataSet, featIndex, value)
            if len(subDataSet) == 0:
                retNode.children[value] = TreeNode(calMostExamplesClass(examples), True)
            else:
                retNode.children[value] = self.createTree(subDataSet, new_labels, new_attrValues)
        return retNode
            
def preOrder(DT):
    print (DT.feature, DT.children.keys())
    if DT.isleaf:
        return
    for item in DT.children.values():
        preOrder(item)
    
        
if __name__ == '__main__':
    calMostExamplesClass([1,1,2,2,2])
    f = open('test.txt')
    dataSet=[]
    labels=[]
    s = f.readline()
    labels.extend(s.strip().split('\t'))
    for line in f:
        dataSet.append(line.strip().split())
    f.close()
    #print(calEntropy(dataSet))
    dt = DecisionTree(dataSet, labels, 'gini')
    preOrder(dt.root)
        
