import matplotlib.pyplot as plt

decisionNode = dict(boxstyle="sawtooth", fc="0.8")    #定义文本框和箭头格式
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

def plotNode(nodeText, centerPt, parentPt, nodeType): #绘制带箭头的注解
    createPlot.ax1.annotate(nodeText, xy=parentPt, xycoords="axes fraction", xytext=centerPt, textcoords="axes fraction",
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)

def createPlot():
    fig = plt.figure(1, facecolor="white")
    fig.clf()
    createPlot.ax1 = plt.subplot(111, frameon=False)
    plotNode("decisionNode", (0.5, 0.1), (0.1, 0.5), decisionNode)
    plotNode("leafNode", (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()

def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]) == dict:           #测试节点的数据类型是否为字典
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]) == dict:
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        maxDepth = max(maxDepth, thisDepth)
    return maxDepth

def retrieveTree(i):
    listOfTrees = [{'no surfacing':{0:'no', 1:{'flippers': {0:'no', 1:'yes'}}}},
                   {'no surfacing':{0:'no', 1:{'flippers': {0:{'head':{0:'no', 1:'yes'}}, 1:'no'}}}} ]
    return listOfTrees[i]

def plotMidText(cntrPt, parentPt, txtSpring):           #在父子节点间填充文本信息
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    ymid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, ymid, txtSpring)
    
def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)                      #计算宽与高
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)              #标记子节点属性值
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD #减少y偏移
    for key in secondDict.keys():
        if type(secondDict[key]) == dict:
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD

def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()

#createPlot()

#print(retrieveTree(1))
#myTree = retrieveTree(0)
#print(getNumLeafs(myTree))
#print(getTreeDepth(myTree))

#myTree = retrieveTree(0)
#createPlot(myTree)
#myTree['no surfacing'][3] = 'maybe'
#print(myTree)
#createPlot(myTree)

