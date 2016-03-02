import matplotlib.pyplot as plt
from functools import reduce

from pylab import *

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction', xytext=centerPt, textcoords='axes fraction' \
                            , ha="center", va="center", bbox=nodeType, arrowprops=arrow_args, size=20)


def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString)


def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = countLeaf(myTree)
    depth = getHeight(myTree)
    firstStr = myTree.feature
    cntrPt = (plotTree.xOff + (1.0 + numLeafs) / 2 / plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD

    for key, item in myTree.children.items():
        if not item.isleaf:
            plotTree(item, cntrPt, key)
        else:
            plotTree.xOff = plotTree.xOff + 1 / plotTree.totalW
            plotNode(item.feature, (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, key)
    plotTree.yOff = plotTree.yOff + 1 / plotTree.totalD
    # print (plotTree.yOff)


def countLeaf(DT):
    if DT.isleaf:
        return 1
    return reduce(lambda x, y: x + y, map(countLeaf, [node for node in DT.children.values()]))


def getHeight(DT):
    if DT.isleaf:
        return 1
    return reduce(lambda x, y: max(x, y), map(getHeight, [node for node in DT.children.values()])) + 1


def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = countLeaf(inTree)
    plotTree.totalD = getHeight(inTree)
    plotTree.xOff = - 0.5 / plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()
